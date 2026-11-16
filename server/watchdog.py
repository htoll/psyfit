"""
PsyFit self-hosting watchdog.

Runs ``streamlit run app.py`` as a child process and keeps it healthy:

* Priority: the server runs at below-normal CPU priority and low I/O priority, with
  numpy/BLAS capped at a few threads, so anything the desktop's owner runs wins.
* Crashes / hangs: restarted when the process exits or the health endpoint stops
  answering, with backoff so a broken app can't spin in a tight crash loop.
* Memory: restarted if it grows past a hard limit (the app itself clears its caches at
  a lower soft limit first; see server_runtime.py).
* Nightly reset: restarted once a night while nobody is using it, which clears every
  cache and picks up code or package changes.
* Sleep: keeps the PC from going to sleep while the server runs (the display can still
  turn off and the screen can still lock).

Started by the "PsyFit server" scheduled task (see install.ps1), which also re-launches
this script if it ever dies. Only one copy runs at a time.

Control files in server/data/:  restart.request  → restart the app now
                                stop.request     → stop the app and this watchdog
"""
import ctypes
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "server" / "data"
PYTHON = REPO / "venv" / "Scripts" / "python.exe"

PORT = int(os.environ.get("PSYFIT_PORT", "8501"))
MEM_HARD_GB = float(os.environ.get("PSYFIT_MEM_HARD_GB", "16"))   # PC has 32 GB
NIGHTLY_WINDOW = (3, 6)            # restart between 03:00 and 06:00 ...
NIGHTLY_IDLE_S = 30 * 60           # ... once nobody has used the app for 30 min
CHECK_EVERY_S = 15
STARTUP_GRACE_S = 120              # health checks start after this
HEALTH_FAILS_BEFORE_RESTART = 4    # = ~1 min unresponsive
BLAS_THREADS = "4"                 # of 12 hardware threads

LOCK = DATA / "watchdog.pid"
STATUS = DATA / "status.json"      # written by the app (server_runtime.py)
RESTART_FLAG = DATA / "restart.request"
STOP_FLAG = DATA / "stop.request"
LOG = DATA / "watchdog.log"
APP_LOG = DATA / "streamlit.log"

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def log(msg: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    _rotate(LOG, 5_000_000)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")


def _rotate(path: Path, max_bytes: int) -> None:
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            os.replace(path, path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass


def _already_running() -> bool:
    """True if another live watchdog holds the lock file."""
    try:
        pid = int(LOCK.read_text().strip())
        p = psutil.Process(pid)
        return pid != os.getpid() and "watchdog.py" in " ".join(p.cmdline())
    except (OSError, ValueError, psutil.Error):
        return False


def _last_activity() -> float:
    try:
        return float(json.loads(STATUS.read_text())["last_activity"])
    except (OSError, ValueError, KeyError):
        return 0.0


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _warn_if_port_taken() -> None:
    """Windows lets a second server bind a port an older Streamlit already holds, and the
    older one keeps receiving the connections, so a forgotten dev instance silently
    shadows this server. Say so in the log rather than starting blind."""
    try:
        holders = {c.pid for c in psutil.net_connections("tcp")
                   if c.laddr and c.laddr.port == PORT and c.status == psutil.CONN_LISTEN}
    except psutil.Error:
        return
    for pid in holders:
        try:
            cmd = " ".join(psutil.Process(pid).cmdline())[:120]
        except psutil.Error:
            cmd = "?"
        log(f"WARNING: port {PORT} is already in use by pid {pid} ({cmd}). Browsers may "
            f"reach that process instead of this server; stop it.")


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc] + proc.children(recursive=True):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


class App:
    def __init__(self):
        self.proc = None
        self.started = 0.0
        self.fails = 0

    def start(self) -> None:
        _warn_if_port_taken()
        env = dict(os.environ, PSYFIT_SERVER="1", OMP_NUM_THREADS=BLAS_THREADS,
                   OPENBLAS_NUM_THREADS=BLAS_THREADS, MKL_NUM_THREADS=BLAS_THREADS,
                   PYTHONUNBUFFERED="1")
        _rotate(APP_LOG, 20_000_000)
        out = open(APP_LOG, "a", encoding="utf-8")
        out.write(f"\n===== start {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        out.flush()
        cmd = [str(PYTHON), "-m", "streamlit", "run", str(REPO / "app.py"),
               "--server.port", str(PORT), "--server.address", "0.0.0.0",
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
        flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
        popen = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT,
                                 creationflags=flags)
        out.close()
        self.proc = psutil.Process(popen.pid)
        try:
            # The venv's python.exe is a launcher; the real interpreter is its child and
            # inherits the CPU priority class, but I/O priority has to be set on each.
            time.sleep(3)
            for p in [self.proc] + self.proc.children(recursive=True):
                p.ionice(psutil.IOPRIO_LOW)
        except psutil.Error:
            pass
        self.started, self.fails = time.time(), 0
        log(f"started app (pid {popen.pid}) on port {PORT}")

    def stop(self, why: str) -> None:
        if self.proc is None:
            return
        log(f"stopping app: {why}")
        try:
            procs = [self.proc] + self.proc.children(recursive=True)
        except psutil.Error:
            procs = [self.proc]
        for p in procs:
            try:
                p.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=15)
        for p in alive:
            try:
                p.kill()
            except psutil.Error:
                pass
        self.proc = None

    def alive(self) -> bool:
        try:
            return self.proc is not None and self.proc.is_running() and \
                self.proc.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            return False


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if _already_running():
        return
    LOCK.write_text(str(os.getpid()))
    STOP_FLAG.unlink(missing_ok=True)
    # Keep the PC awake for as long as this thread lives; cleared automatically on exit.
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    log(f"watchdog up (pid {os.getpid()})")

    app = App()
    backoff = 10.0
    last_nightly = None          # date of the last nightly restart
    app.start()

    while True:
        time.sleep(CHECK_EVERY_S)
        now = time.time()

        if STOP_FLAG.exists():
            STOP_FLAG.unlink(missing_ok=True)
            app.stop("stop requested")
            log("watchdog exiting")
            LOCK.unlink(missing_ok=True)
            return

        if RESTART_FLAG.exists():
            RESTART_FLAG.unlink(missing_ok=True)
            app.stop("restart requested from admin page")
            app.start()
            continue

        if not app.alive():
            uptime = now - app.started
            log(f"app exited after {uptime:.0f} s; restarting in {backoff:.0f} s")
            time.sleep(backoff)
            # Grow the delay only for crash loops; a long healthy run resets it.
            backoff = 10.0 if uptime > 600 else min(backoff * 2, 300.0)
            app.start()
            continue

        if now - app.started > STARTUP_GRACE_S:
            app.fails = 0 if _healthy() else app.fails + 1
            if app.fails >= HEALTH_FAILS_BEFORE_RESTART:
                app.stop("health check failing")
                app.start()
                continue

        try:
            rss = _tree_rss(app.proc)
        except psutil.Error:
            rss = 0
        if rss > MEM_HARD_GB * 1e9:
            app.stop(f"memory {rss / 1e9:.1f} GB over the {MEM_HARD_GB:g} GB hard limit")
            app.start()
            continue

        today = datetime.now().date()
        hour = datetime.now().hour
        if (NIGHTLY_WINDOW[0] <= hour < NIGHTLY_WINDOW[1] and last_nightly != today
                and now - max(_last_activity(), app.started) > NIGHTLY_IDLE_S):
            last_nightly = today
            app.stop("nightly reset")
            app.start()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("watchdog crashed:\n" + traceback.format_exc())
        LOCK.unlink(missing_ok=True)
        sys.exit(1)
