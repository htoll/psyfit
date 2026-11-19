# server_runtime.py
"""
Self-hosted server support for PsyFit: live presence, usage analytics, saving files to
the lab drive, and memory housekeeping.

Everything here is inert unless the app was started by server/watchdog.py, which sets
PSYFIT_SERVER=1. A plain ``streamlit run app.py`` (or Streamlit Cloud) behaves exactly
as before.

How it hooks in: app.py calls :func:`begin_run` at the top of every rerun. The first
call wraps ``file_uploader`` and ``download_button`` so every upload and download in
every tool is logged (and optionally copied to the lab drive) without touching the
tools themselves.

State lives in server/data/ (gitignored): usage.db (SQLite: events + per-computer
preferences) and status.json (last activity, read by the watchdog to pick a quiet
moment for its nightly restart).
"""
import functools
import gc
import inspect
import json
import os
import queue
import re
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import streamlit as st

ENABLED = os.environ.get("PSYFIT_SERVER") == "1"

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "server" / "data"
DB_PATH = DATA_DIR / "usage.db"
STATUS_PATH = DATA_DIR / "status.json"
RESTART_FLAG = DATA_DIR / "restart.request"
SYNC_ROOT = Path(os.environ.get("PSYFIT_SYNC_ROOT", r"G:\Shared drives\PengLab_Data_2026"))
# Clear Streamlit's data caches once the server process grows past this. The watchdog
# restarts the whole process at a higher hard limit if that isn't enough.
MEM_SOFT_LIMIT_GB = float(os.environ.get("PSYFIT_MEM_SOFT_GB", "10"))
STARTED_AT = time.time()

_lock = threading.RLock()
_sessions = {}        # session_id -> {ip, user, tool, first_seen, last_seen}
_sync_msgs = {}       # session_id -> [(ts, message, ok)], newest last
_dir_cache = {}       # relative dir -> (ts, [subfolder names])
_last_status_write = 0.0
_last_mem_check = 0.0
_patched = False

# Session-state keys (all prefixed so they can't collide with a tool's own keys).
K_USER, K_SYNC_ON, K_SYNC_DIR = "srv_user", "srv_sync_on", "srv_sync_dir"
K_SYNC_UP, K_SYNC_DOWN = "srv_sync_up", "srv_sync_down"
K_SEEN = "srv_seen_uploads"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    ts REAL, day TEXT, session TEXT, ip TEXT, user TEXT, tool TEXT,
    event TEXT, detail TEXT, bytes INTEGER);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS prefs (
    ip TEXT PRIMARY KEY, user TEXT, sync_on INTEGER, sync_dir TEXT,
    sync_up INTEGER, sync_down INTEGER, updated REAL);
"""
_db_ready = False


def _connect() -> sqlite3.Connection:
    global _db_ready
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    if not _db_ready:
        con.executescript(_SCHEMA)
        _db_ready = True
    return con


def _execute(sql: str, params=()) -> None:
    with _lock, closing(_connect()) as con, con:
        con.execute(sql, params)


def query(sql: str, params=()):
    """Run a read query against the usage database; returns a DataFrame."""
    import pandas as pd
    if not DB_PATH.exists():
        return pd.DataFrame()
    with _lock, closing(_connect()) as con:
        return pd.read_sql_query(sql, con, params=params)


def log_event(event: str, detail: str = "", nbytes=None, sid=None) -> None:
    """Record one usage event. Safe to call from any thread; never raises."""
    if not ENABLED:
        return
    sid = sid or _session_id()
    info = _sessions.get(sid, {})
    now = time.time()
    try:
        _execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)",
                 (now, datetime.fromtimestamp(now).strftime("%Y-%m-%d"), sid,
                  info.get("ip"), info.get("user") or None, info.get("tool"),
                  event, str(detail)[:500], nbytes))
    except Exception:
        pass


def _load_prefs(ip: str) -> dict:
    try:
        with _lock, closing(_connect()) as con:
            row = con.execute("SELECT user, sync_on, sync_dir, sync_up, sync_down "
                              "FROM prefs WHERE ip = ?", (ip,)).fetchone()
    except Exception:
        row = None
    if not row:
        return {}
    return {K_USER: row[0] or "", K_SYNC_ON: bool(row[1]), K_SYNC_DIR: row[2] or "",
            K_SYNC_UP: bool(row[3]), K_SYNC_DOWN: bool(row[4])}


def _save_prefs() -> None:
    """Remember this computer's name and sync settings (keyed by IP address)."""
    ss = st.session_state
    try:
        _execute("INSERT OR REPLACE INTO prefs VALUES (?,?,?,?,?,?,?)",
                 (_client_ip(), ss.get(K_USER, ""), int(bool(ss.get(K_SYNC_ON))),
                  ss.get(K_SYNC_DIR, ""), int(bool(ss.get(K_SYNC_UP, True))),
                  int(bool(ss.get(K_SYNC_DOWN, True))), time.time()))
    except Exception:
        pass
    sid = _session_id()
    if sid in _sessions:
        _sessions[sid]["user"] = ss.get(K_USER, "")


# ---------------------------------------------------------------------------
# Sessions and presence
# ---------------------------------------------------------------------------
def _session_id():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else None
    except Exception:
        return None


def _client_ip() -> str:
    """The browser's IP; the server computer is always "localhost", however it connected."""
    try:
        ip = st.context.ip_address
    except Exception:
        return "unknown"
    return "localhost" if ip is None or ip in _server_ips() else ip


_own_ips = (0.0, frozenset())


def _server_ips() -> frozenset:
    """This computer's own network addresses (refreshed every 5 min, as Wi-Fi/DHCP can
    change them). Opening the app via the PC's LAN address instead of localhost makes
    Streamlit report that address rather than None."""
    global _own_ips
    if time.time() - _own_ips[0] > 300:
        try:
            import psutil
            ips = {a.address.split("%")[0] for addrs in psutil.net_if_addrs().values()
                   for a in addrs}
        except Exception:
            ips = set()
        _own_ips = (time.time(), frozenset(ips))
    return _own_ips[1]


def is_local_request() -> bool:
    """True when the browser is on the server computer itself."""
    try:
        ip = st.context.ip_address
    except Exception:
        return False
    return ip is None or ip in _server_ips()


def _connected_session_ids():
    """IDs of sessions with an open browser connection, or None if unavailable.
    Uses a private Streamlit API, so it is guarded; presence then falls back to
    'seen in the last 10 minutes'."""
    try:
        from streamlit import runtime
        mgr = runtime.get_instance()._session_mgr
        return {s.session.id for s in mgr.list_active_sessions()}
    except Exception:
        return None


def _prune_sessions() -> None:
    live = _connected_session_ids()
    cutoff = time.time() - 600
    with _lock:
        for sid in list(_sessions):
            if (live is not None and sid not in live) or \
                    (live is None and _sessions[sid]["last_seen"] < cutoff):
                del _sessions[sid]
                _sync_msgs.pop(sid, None)


def live_sessions() -> list:
    """Currently connected sessions, for the admin page."""
    _prune_sessions()
    with _lock:
        return [dict(sid=sid, **info) for sid, info in _sessions.items()]


def _people_online() -> set:
    _prune_sessions()
    with _lock:
        return {info["ip"] for info in _sessions.values()}


@st.fragment(run_every=30)
def _presence_line() -> None:
    people = _people_online()
    others = len(people - {_client_ip()})
    word = "person" if others == 1 else "people"
    st.caption(f":green[●] {others} other {word} using PsyFit right now")


def render_presence() -> None:
    if ENABLED:
        with st.sidebar:
            _presence_line()


# ---------------------------------------------------------------------------
# Per-rerun hook
# ---------------------------------------------------------------------------
def begin_run(tool: str) -> None:
    """Call at the top of every app rerun with the selected tool's label."""
    if not ENABLED:
        return
    _install_patches()
    sid = _session_id()
    if sid is None:
        return
    now = time.time()
    with _lock:
        info = _sessions.get(sid)
        is_new = info is None
        if is_new:
            info = _sessions[sid] = {"ip": _client_ip(), "user": "", "tool": None,
                                     "first_seen": now}
        info["last_seen"] = now
        prev_tool, info["tool"] = info["tool"], tool

    ss = st.session_state
    if "srv_prefs_loaded" not in ss:
        defaults = {K_USER: "", K_SYNC_ON: False, K_SYNC_DIR: "",
                    K_SYNC_UP: True, K_SYNC_DOWN: True}
        defaults.update(_load_prefs(_client_ip()))
        for k, v in defaults.items():
            ss.setdefault(k, v)
        ss["srv_prefs_loaded"] = True
    info["user"] = ss.get(K_USER, "")

    if is_new:
        try:
            agent = st.context.headers.get("User-Agent", "")
        except Exception:
            agent = ""
        log_event("session_start", agent[:200])
    if tool != prev_tool:
        log_event("open_tool", tool)
    _write_status(now)
    _memory_guard(now)


def _write_status(now: float) -> None:
    global _last_status_write
    if now - _last_status_write < 30:
        return
    _last_status_write = now
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps({"last_activity": now, "sessions": len(_sessions),
                                   "pid": os.getpid()}))
        os.replace(tmp, STATUS_PATH)
    except OSError:
        pass


def process_memory_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def clear_caches(reason: str) -> None:
    st.cache_data.clear()
    gc.collect()
    log_event("cache_clear", reason)


def _memory_guard(now: float) -> None:
    global _last_mem_check
    if now - _last_mem_check < 60:
        return
    _last_mem_check = now
    gb = process_memory_gb()
    if gb > MEM_SOFT_LIMIT_GB:
        clear_caches(f"memory {gb:.1f} GB over the {MEM_SOFT_LIMIT_GB:g} GB soft limit")


def request_restart() -> None:
    """Ask the watchdog to restart the app (picked up within ~15 s)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESTART_FLAG.write_text(str(time.time()))


# ---------------------------------------------------------------------------
# Upload / download hooks
# ---------------------------------------------------------------------------
def _install_patches() -> None:
    """Wrap file_uploader and download_button once per process. ``st.x`` is a method
    bound to the main container at import time, so it is wrapped separately from the
    DeltaGenerator class (which covers st.sidebar.x and column/expander containers)."""
    global _patched
    if _patched:
        return
    from streamlit.delta_generator import DeltaGenerator
    for name, after in (("file_uploader", _after_upload), ("download_button", _after_download)):
        setattr(DeltaGenerator, name, _wrap(getattr(DeltaGenerator, name), after))
        setattr(st, name, _wrap(getattr(st, name), after))
    _patched = True


def _wrap(orig, after):
    try:
        sig = inspect.signature(orig)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        result = orig(*args, **kwargs)
        try:
            bound = sig.bind_partial(*args, **kwargs).arguments if sig else kwargs
            after(result, bound)
        except Exception:
            pass   # logging/sync must never break a tool
        return result
    return wrapper


def _after_upload(result, _args) -> None:
    files = result if isinstance(result, list) else ([result] if result is not None else [])
    seen = st.session_state.setdefault(K_SEEN, set())
    new = [f for f in files if getattr(f, "file_id", id(f)) not in seen]
    if not new:
        return
    seen.update(getattr(f, "file_id", id(f)) for f in new)
    log_event("upload", ", ".join(f.name for f in new),
              sum(int(getattr(f, "size", 0)) for f in new))
    if _sync_wanted(K_SYNC_UP):
        for f in new:
            _queue_write(f.name, f.getvalue, "upload")


def _after_download(clicked, args) -> None:
    if not clicked:
        return
    data, name = args.get("data"), args.get("file_name") or "download"
    payload = _to_bytes(data)
    log_event("download", name, len(payload) if payload is not None else None)
    if payload is not None and _sync_wanted(K_SYNC_DOWN):
        _queue_write(name, lambda: payload, "download")


def _to_bytes(data):
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    if hasattr(data, "getvalue"):
        return data.getvalue()
    return None     # deferred (callable) data: don't evaluate it a second time


# ---------------------------------------------------------------------------
# Lab-drive sync
# ---------------------------------------------------------------------------
_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_write_q = queue.Queue()
_worker_started = False


def _sync_wanted(kind_key: str) -> bool:
    ss = st.session_state
    return bool(ss.get(K_SYNC_ON) and ss.get(kind_key) and SYNC_ROOT.is_dir())


def _target_dir(rel: str) -> str:
    """Absolute folder for a path relative to SYNC_ROOT; refuses anything outside it."""
    root = os.path.normpath(str(SYNC_ROOT))
    path = os.path.normpath(os.path.join(root, rel or ""))
    if os.path.commonpath([root.lower(), path.lower()]) != root.lower():
        raise ValueError(f"{rel!r} is outside the lab drive")
    return path


def _unique_path(folder: str, name: str, size: int):
    """Where to write ``name``; None if an identical-size file is already there.
    Never overwrites: a different file with the same name gets ' (2)', ' (3)', ..."""
    stem, ext = os.path.splitext(name)
    for i in range(1, 1000):
        cand = os.path.join(folder, name if i == 1 else f"{stem} ({i}){ext}")
        if not os.path.exists(cand):
            return cand
        if os.path.getsize(cand) == size:
            return None
    raise OSError("too many files with the same name")


def _queue_write(name: str, getter, kind: str) -> None:
    global _worker_started
    if not _worker_started:
        threading.Thread(target=_sync_worker, daemon=True, name="psyfit-sync").start()
        _worker_started = True
    safe = _INVALID.sub("_", os.path.basename(str(name))).strip() or "file"
    _write_q.put((_session_id(), st.session_state.get(K_SYNC_DIR, ""), safe, getter, kind))


def _sync_worker() -> None:
    """Writes queued files one at a time, off the script thread, so a large upload
    doesn't stall the tool while it is copied to the drive."""
    while True:
        sid, rel, name, getter, kind = _write_q.get()
        shown = f"{rel}/{name}" if rel else name
        try:
            data = getter()
            folder = _target_dir(rel)
            os.makedirs(folder, exist_ok=True)
            path = _unique_path(folder, name, len(data))
            if path is None:
                msg, ok = f"{shown} is already on the drive", True
            else:
                tmp = path + ".partial"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
                saved = os.path.relpath(path, str(SYNC_ROOT)).replace(os.sep, "/")
                msg, ok = f"Saved {saved}", True
                log_event("sync", f"{kind}: {saved}", len(data), sid=sid)
        except Exception as e:
            msg, ok = f"Couldn't save {shown}: {e}", False
            log_event("sync_error", msg, sid=sid)
        with _lock:
            lst = _sync_msgs.setdefault(sid, [])
            lst.append((time.time(), msg, ok))
            del lst[:-8]


def _subfolders(rel: str) -> list:
    now = time.time()
    hit = _dir_cache.get(rel)
    if hit and now - hit[0] < 60:
        return hit[1]
    try:
        with os.scandir(_target_dir(rel)) as it:
            names = sorted((e.name for e in it if e.is_dir() and not e.name.startswith(".")),
                           key=str.lower)
    except (OSError, ValueError):
        names = []
    _dir_cache[rel] = (now, names)
    return names


def _enter_subfolder() -> None:
    pick = st.session_state.get("srv_sync_pick")
    if pick and pick != _PICK_NONE:
        cur = st.session_state.get(K_SYNC_DIR, "")
        st.session_state[K_SYNC_DIR] = f"{cur}/{pick}" if cur else pick
        _save_prefs()
    st.session_state["srv_sync_pick"] = _PICK_NONE


def _go_up() -> None:
    cur = st.session_state.get(K_SYNC_DIR, "")
    st.session_state[K_SYNC_DIR] = cur.rsplit("/", 1)[0] if "/" in cur else ""
    _save_prefs()


def _create_subfolder() -> None:
    raw = (st.session_state.get("srv_sync_new") or "").strip()
    name = _INVALID.sub("_", raw).strip(" .")
    st.session_state["srv_sync_new"] = ""
    if not name:
        return
    cur = st.session_state.get(K_SYNC_DIR, "")
    rel = f"{cur}/{name}" if cur else name
    try:
        os.makedirs(_target_dir(rel), exist_ok=True)
    except (OSError, ValueError) as e:
        st.session_state["srv_sync_err"] = f"Couldn't create {name}: {e}"
        return
    _dir_cache.pop(cur, None)
    st.session_state[K_SYNC_DIR] = rel
    _save_prefs()


_PICK_NONE = "(choose a subfolder)"


def render_sync_panel() -> None:
    """Sidebar panel: your name, and where (if anywhere) to copy your files."""
    if not ENABLED:
        return
    ss = st.session_state
    with st.sidebar.expander("Save files to lab drive", expanded=False):
        st.text_input("Your name", key=K_USER, on_change=_save_prefs,
                      help="Optional. Lets the server admin see who is using which tools.")
        if not SYNC_ROOT.is_dir():
            st.caption("The lab drive isn't reachable from the server right now.")
            return
        st.toggle("Copy my files to the lab drive", key=K_SYNC_ON, on_change=_save_prefs,
                  help=f"Copies what you upload and/or download into a folder of your choice "
                       f"inside {SYNC_ROOT}. Existing files are never overwritten.")
        if ss.get(K_SYNC_ON):
            rel = ss.get(K_SYNC_DIR, "")
            st.markdown(f"**Folder:** `{SYNC_ROOT.name}/{rel}`")
            subs = _subfolders(rel)
            if subs:
                st.selectbox("Open a subfolder", [_PICK_NONE] + subs, key="srv_sync_pick",
                             on_change=_enter_subfolder)
            c1, c2 = st.columns(2)
            c1.button("Up one level", on_click=_go_up, disabled=not rel,
                      use_container_width=True)
            with c2.popover("New folder", use_container_width=True):
                st.text_input("Folder name", key="srv_sync_new")
                st.button("Create and open", on_click=_create_subfolder)
            if ss.get("srv_sync_err"):
                st.error(ss.pop("srv_sync_err"))
            st.checkbox("Uploaded files", key=K_SYNC_UP, on_change=_save_prefs)
            st.checkbox("Downloaded results", key=K_SYNC_DOWN, on_change=_save_prefs)
            if not rel:
                st.warning("Files will land directly in the top-level folder. "
                           "Pick a subfolder above.")
            with _lock:
                msgs = list(_sync_msgs.get(_session_id(), []))
            for _, msg, ok in reversed(msgs[-4:]):
                st.caption(("✓ " if ok else "✗ ") + msg)
        st.caption("Settings are remembered for this computer. Tool use and file names "
                   "are logged for the server admin.")
