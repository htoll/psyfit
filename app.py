# app.py
import os
import sys
import json
import traceback
import importlib
import importlib.util
from importlib import metadata as importlib_metadata
import platform
import subprocess
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components
from zoneinfo import ZoneInfo

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
# Ensure local imports work when running "streamlit run app.py"
sys.path.insert(0, REPO_ROOT)


def _repo_last_updated(repo_path: str) -> str:
    """Return a human-readable timestamp for the last git commit in ``repo_path``."""
    try:
        timestamp_raw = subprocess.check_output(
            ["git", "-C", repo_path, "log", "-1", "--format=%ct"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"

    if not timestamp_raw:
        return "unknown"

    try:
        timestamp = datetime.fromtimestamp(
            int(timestamp_raw), tz=timezone.utc
        ).astimezone(ZoneInfo("America/New_York"))
    except (ValueError, OSError, OverflowError):
        return "unknown"

    return timestamp.strftime("%Y-%m-%d %H:%M %Z")


# --- Page setup ---
try:
    st.set_page_config(
        page_title="PsyFit",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass


# ---------------------------------------------------------------------------
# Dynamic browser-tab title
# ---------------------------------------------------------------------------
# ``set_page_config`` only sets a static title, so we drive the tab text live
# with a small JS hook injected via components.html (which can reach the parent
# document). One persistent hook — installed once — shows:
#   • "Uploading files…"  while a file input is receiving files
#   • "Analyzing…"        while Streamlit is running (its status widget is shown)
#   • the tool name        otherwise (idle), shortened to fit a narrow tab
# We only re-send the current idle name each rerun; the hook does the switching.

# Compact tab labels; anything not listed falls back to truncation.
SHORT_TOOL_NAMES = {
    "Brightness (WF)": "Brightness WF",
    "Brightness (Conf)": "Brightness Conf",
    "Movie Brightness": "Movie Bright",
    "Saturation Series": "Sat Series",
    "Confocal Visualization": "Confocal Viz",
    "Dye Colocalization": "Coloc",
    "Process Spectra": "Proc Spectra",
    "Monomer Estimation": "Monomers",
    "Shelling Injection Table": "Shelling",
    "TEM Size Analysis": "TEM Size",
    "FFT Analysis": "FFT",
}


def _short_tab_name(label: str, maxlen: int = 18) -> str:
    name = SHORT_TOOL_NAMES.get(label, label)
    if len(name) > maxlen:
        name = name[: maxlen - 1].rstrip() + "…"
    return name


def _sync_tab_title(idle_name: str):
    """Install (once) the tab-title hook and set the current idle name.

    Uses a single 0-px iframe per rerun. The hook watches Streamlit's running
    indicator (``stStatusWidget``) to show "Analyzing…", a file-input change to
    show "Uploading files…", and otherwise the idle tool name.
    """
    components.html(
        """
        <script>
        (function () {
            const doc = window.parent.document;
            doc.__psyfitIdle = %s;
            const RUNNING = '[data-testid="stStatusWidgetRunningIcon"], [data-testid="stStatusWidgetRunningManIcon"]';
            function apply() {
                if (doc.querySelector(RUNNING)) {
                    doc.title = 'Analyzing…';
                } else if (!(doc.title || '').startsWith('Uploading')) {
                    doc.title = doc.__psyfitIdle;
                }
            }
            if (!doc.__psyfitTitleHook) {
                doc.__psyfitTitleHook = true;
                doc.addEventListener('change', function (e) {
                    const t = e.target;
                    if (t && t.tagName === 'INPUT' && t.type === 'file' && t.files && t.files.length) {
                        doc.title = 'Uploading files…';
                    }
                }, true);
                new MutationObserver(apply).observe(doc.body, {childList: true, subtree: true});
            }
            apply();
        })();
        </script>
        """ % json.dumps(idle_name),
        height=0,
    )


# ---------------------------------------------------------------------------
# Tool registry
# Display Name -> (module_path, callable_name, category, description, beta)
# ---------------------------------------------------------------------------
TOOLS = {
    "Batch Convert": (
        "tools.batch_convert", "run", "Convert & Export",
        "Bulk-convert .sif files into images/data; splits into quadrants with UCNP or dye detection.",
        False,
    ),
    "Process Movie": (
        "tools.read_movie", "run", "Convert & Export",
        "Export .sif movies to MP4/MOV/TIFF with region crops, labels, and colorbars.",
        False,
    ),
    "Brightness (WF)": (
        "tools.analyze_single_sif", "run", "Brightness & Intensity",
        "Detect emitters in widefield .sif images and quantify per-particle brightness.",
        False,
    ),
    "Brightness (Conf)": (
        "tools.confocal_brightness", "run", "Brightness & Intensity",
        "Analyze confocal .dat files for per-particle brightness with tunable fit thresholds.",
        False,
    ),
    "Movie Brightness": (
        "tools.movie_brightness", "run", "Brightness & Intensity",
        "Fit per-PSF brightness across every frame of a .sif movie; per-emitter "
        "and region intensity-vs-time traces. Local-only (disabled on cloud).",
        True,
    ),
    "Saturation Series": (
        "tools.SaturationSeries", "run", "Brightness & Intensity",
        "Plot brightness versus excitation power density across a saturation series by quadrant.",
        False,
    ),
    "Confocal Visualization": (
        "tools.confocal_visualizer", "run", "Visualization",
        "Visualize and merge confocal channels with custom colormaps and grid layouts.",
        False,
    ),
    "Dye Colocalization": (
        "tools.colocalization", "run", "Spectral & Colocalization",
        "Measure colocalization between dye channels across registered images.",
        False,
    ),
    "Get Spectra": (
        "tools.get_spectra", "run", "Spectral & Colocalization",
        "Extract and plot emission spectra from uploaded acquisitions.",
        False,
    ),
    "Process Spectra": (
        "tools.process_spectra", "run", "Spectral & Colocalization",
        "Compare Get Spectra CSVs: per-file plots, click-to-exclude, averaging, normalization, and volume scaling.",
        False,
    ),
    "Monomer Estimation": (
        "tools.monomers", "run", "Quantification",
        "Estimate monomer/dimer/trimer/multimer populations from brightness distributions. "
        "Concentration estimation assumes uniform distribution of particles across 3 mm PDMS "
        "well volume using 5 uL of 1x PBS that has been allowed to equilibrate for > 5 mins.",
        False,
    ),
    "Shelling Injection Table": (
        "tools.shelling_table", "run", "Synthesis",
        "Compute shell-growth injection volumes and timing for nanocrystal synthesis.",
        False,
    ),
    "TEM Size Analysis": (
        "tools.tem_analysis", "run", "TEM",
        "Characterize particle size and shape from TEM images using computer-vision watershed fitting.",
        True,
    ),
    "FFT Analysis": (
        "tools.fft", "run", "TEM",
        "Estimate lattice spacing from TEM images via interactive FFT analysis.",
        True,
    ),
    # "Plot CSVs": ("tools.plot_csv", "run", "Convert & Export", "Flexible CSV plotting.", False),
}

# Preserve registry insertion order for categories.
CATEGORY_ORDER = []
for _label, (_m, _f, _cat, _d, _b) in TOOLS.items():
    if _cat not in CATEGORY_ORDER:
        CATEGORY_ORDER.append(_cat)


# ---------------------------------------------------------------------------
# Sidebar: navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🔬 PsyFit")
st.sidebar.caption("Microscopy & nanoparticle analysis toolkit")

category = st.sidebar.radio(
    "Category",
    CATEGORY_ORDER,
    index=0,
)

tools_in_category = [
    label for label, (_m, _f, cat, _d, _b) in TOOLS.items() if cat == category
]


def _format_tool(label: str) -> str:
    is_beta = TOOLS[label][4]
    return f" {label}" if is_beta else label


tool_label = st.sidebar.radio(
    "Tool",
    tools_in_category,
    index=0,
    format_func=_format_tool,
)

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar: settings & diagnostics (tucked at the bottom)
# ---------------------------------------------------------------------------
show_traces = st.sidebar.toggle(
    "Show error tracebacks",
    value=False,
    help="Expand to see full Python tracebacks when tools error.",
)

with st.sidebar.expander("Diagnostics", expanded=False):
    st.markdown("**Environment**")
    st.code(
        f"python: {platform.python_version()}\n"
        f"os: {platform.system()} {platform.release()}\n"
        f"executable: {sys.executable}\n"
        f"cwd: {os.getcwd()}",
        language="bash",
    )

    wanted_pkgs = [
        "numpy", "scipy", "scikit-image", "scikit-learn",
        "matplotlib", "pandas", "streamlit",
    ]

    def pkg_version(name: str) -> str:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return "not installed"
        except Exception as e:
            return f"error: {type(e).__name__}"

    st.markdown("**Package versions**")
    st.code(
        "\n".join(f"{p}: {pkg_version(p)}" for p in wanted_pkgs),
        language="bash",
    )

    st.markdown("**Tool availability (light check)**")
    rows = []
    for label, (modpath, _f, _c, _d, _b) in TOOLS.items():
        spec = importlib.util.find_spec(modpath)
        rows.append(f"{label}: {'found' if spec else 'MISSING'}")
    st.code("\n".join(rows), language="bash")

    st.markdown("**Deep check a tool** (imports module)")
    deep_tool = st.selectbox("Pick a tool to deep-check:", list(TOOLS.keys()))
    if st.button("Run deep check"):
        modpath, funcname = TOOLS[deep_tool][0], TOOLS[deep_tool][1]
        try:
            module = importlib.import_module(modpath)
            fn = getattr(module, funcname, None)
            st.success(
                f"Imported `{modpath}` OK. "
                f"{'Found' if fn else 'Missing'} `{funcname}`; "
                f"{'callable' if callable(fn) else 'not callable'}."
            )
        except Exception as e:
            st.error(f"Deep check failed for {deep_tool}: {type(e).__name__}: {e}")
            if show_traces:
                st.code("".join(traceback.format_exception(e)), language="pytb")

st.sidebar.caption(f"Last repository update: {_repo_last_updated(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main area: header + selected tool
# ---------------------------------------------------------------------------
def render_error_context(title: str, err: Exception):
    st.error(f"⚠️ {title}: {type(err).__name__}: {err}")
    if show_traces:
        tb = "".join(traceback.format_exception(err))
        with st.expander("View traceback"):
            st.code(tb, language="pytb")
    else:
        st.caption("Enable *Show error tracebacks* in the sidebar for full details.")


def safe_import(module_path: str):
    try:
        return importlib.import_module(module_path), None
    except Exception as e:
        return None, e


def safe_getattr(module, attr: str):
    try:
        fn = getattr(module, attr)
        if not callable(fn):
            raise TypeError(f"Attribute '{attr}' on '{module.__name__}' is not callable.")
        return fn, None
    except Exception as e:
        return None, e


def safe_run_tool(modpath: str, funcname: str, label: str):
    with st.spinner(f"Loading {label}…"):
        module, import_err = safe_import(modpath)
        if import_err:
            render_error_context(f"Failed to import {label} ({modpath})", import_err)
            return

    run_fn, getattr_err = safe_getattr(module, funcname)
    if getattr_err:
        render_error_context(f"Failed to find '{funcname}()' in {modpath}", getattr_err)
        return

    with st.spinner(f"Running {label}…"):
        try:
            return run_fn()
        except Exception as e:
            render_error_context(f"{label} crashed while running", e)
            return


if tool_label in TOOLS:
    modpath, funcname, cat, description, is_beta = TOOLS[tool_label]

    # Idle tab title = the (shortened) tool name; the hook flips it to
    # "Analyzing…" while running and "Uploading files…" during uploads.
    _sync_tab_title(_short_tab_name(tool_label))

    title = f"{tool_label}" + ("  Beta" if is_beta else "")
    st.title(title)
    st.caption(f"{cat} · {description}")

    st.divider()

    safe_run_tool(modpath, funcname, tool_label)
else:
    _sync_tab_title("PsyFit")
    st.title("🔬 PsyFit")
    st.info("Select a tool from the sidebar to begin.")
