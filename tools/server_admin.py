# tools/server_admin.py
"""
Server Admin: who is using the self-hosted PsyFit server and what they do with it.

Only listed in the navigation when the server runs under server/watchdog.py and the
browser is on the server computer itself (localhost); run() re-checks the latter.
"""
import time
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import server_runtime as srv

RANGES = {"Today": 0, "Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": None}


def _fmt_bytes(n) -> str:
    if n is None or pd.isna(n) or n <= 0:
        return ""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{int(n)} B"


def _ago(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{seconds // 60} min"
    return f"{seconds / 3600:.1f} h"


def _live_section() -> None:
    st.subheader("Right now")
    now = time.time()
    sessions = srv.live_sessions()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("People online", len({s["ip"] for s in sessions}))
    m2.metric("Open tabs", len(sessions))
    m3.metric("Server memory", f"{srv.process_memory_gb():.1f} GB",
              help=f"Caches are cleared automatically above {srv.MEM_SOFT_LIMIT_GB:g} GB; "
                   "the watchdog restarts the server at its hard limit.")
    m4.metric("Up for", _ago(now - srv.STARTED_AT))
    if sessions:
        st.dataframe(pd.DataFrame([{
            "Name": s["user"] or "", "Computer (IP)": s["ip"], "Tool": s["tool"],
            "Connected for": _ago(now - s["first_seen"]),
            "Last action": _ago(now - s["last_seen"]) + " ago",
        } for s in sessions]), hide_index=True, use_container_width=True)

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Clear caches now"):
        srv.clear_caches("manual, from admin page")
        st.success("Caches cleared.")
    if c2.button("Restart server", help="Everyone connected is disconnected for ~20 s."):
        srv.request_restart()
        st.warning("Restart requested. The page reconnects in about 20 seconds.")


def _usage_section() -> None:
    st.subheader("Usage")
    span = st.radio("Period", list(RANGES), index=2, horizontal=True, key="adm_range")
    days = RANGES[span]
    if days is None:
        since = 0.0
    else:
        since = datetime.combine(datetime.now().date() - timedelta(days=days),
                                 datetime.min.time()).timestamp()
    ev = srv.query("SELECT * FROM events WHERE ts >= ? ORDER BY ts", (since,))
    if ev.empty:
        st.info("No usage recorded in this period yet.")
        return

    # A person is their entered name if they gave one, otherwise their computer's IP.
    ev["person"] = ev["user"].where(ev["user"].notna() & (ev["user"] != ""), ev["ip"])
    ev["time"] = pd.to_datetime(ev["ts"], unit="s", utc=True).dt.tz_convert(
        datetime.now().astimezone().tzinfo).dt.strftime("%Y-%m-%d %H:%M")
    up = ev[ev["event"] == "upload"]
    syncs = ev[ev["event"] == "sync"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("People", ev["person"].nunique())
    m2.metric("Visits", int((ev["event"] == "session_start").sum()))
    def _count(df):
        size = _fmt_bytes(df["bytes"].sum())
        return f"{len(df)} ({size})" if size else str(len(df))

    m3.metric("Uploads", _count(up))
    m4.metric("Saved to drive", _count(syncs))
    m5.metric("Tool errors", int((ev["event"] == "error").sum()))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**People per day**")
        st.bar_chart(ev.groupby("day")["person"].nunique().rename("people"))
    with c2:
        st.markdown("**Tool opens**")
        opens = ev[ev["event"] == "open_tool"]
        if not opens.empty:
            st.bar_chart(opens["detail"].value_counts().rename("opens"), horizontal=True)

    st.markdown("**By person**")
    per = ev.groupby("person").agg(
        computer=("ip", lambda s: ", ".join(sorted(set(s.dropna())))),
        visits=("event", lambda s: int((s == "session_start").sum())),
        uploads=("event", lambda s: int((s == "upload").sum())),
        downloads=("event", lambda s: int((s == "download").sum())),
        errors=("event", lambda s: int((s == "error").sum())),
        last_seen=("time", "max"),
    )
    tools = (ev[ev["event"] == "open_tool"].groupby("person")["detail"]
             .agg(lambda s: ", ".join(s.value_counts().index[:5])))
    per["most used tools"] = tools
    st.dataframe(per.sort_values("last_seen", ascending=False), use_container_width=True)

    st.markdown("**Activity log**")
    f1, f2 = st.columns(2)
    who = f1.multiselect("Person", sorted(ev["person"].dropna().unique()), key="adm_who")
    kinds = f2.multiselect("Event", sorted(ev["event"].unique()), key="adm_kind")
    log = ev
    if who:
        log = log[log["person"].isin(who)]
    if kinds:
        log = log[log["event"].isin(kinds)]
    log = log.assign(size=log["bytes"].map(_fmt_bytes))
    st.dataframe(log[["time", "person", "tool", "event", "detail", "size"]].iloc[::-1].head(500),
                 hide_index=True, use_container_width=True)
    st.download_button("Download full log (CSV)", ev.to_csv(index=False).encode("utf-8"),
                       file_name="psyfit_usage.csv", mime="text/csv")


def _logs_section() -> None:
    with st.expander("Watchdog log"):
        path = srv.DATA_DIR / "watchdog.log"
        if path.exists():
            st.code("".join(path.read_text(encoding="utf-8").splitlines(True)[-60:]),
                    language="text")
        else:
            st.caption("No watchdog log yet.")


def run():
    if not srv.ENABLED:
        st.info("Only available when PsyFit runs as the self-hosted server.")
        return
    if not srv.is_local_request():
        st.error("The admin page can only be opened on the server computer.")
        return
    _live_section()
    st.divider()
    _usage_section()
    _logs_section()
