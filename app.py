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


@st.cache_data(show_spinner=False)
def _repo_last_updated(repo_path: str) -> str:
    """Return a human-readable timestamp for the last git commit in ``repo_path``.

    Cached: this shells out to git, and the answer cannot change while the
    process is alive, so it must not run on every rerun.
    """
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
# document). One persistent hook, installed once, shows:
#   "Uploading files..."  while a file input is receiving files
#   "Analyzing..."        while Streamlit is running (its status widget is shown)
#   the tool name         otherwise (idle), shortened to fit a narrow tab
# We only re-send the current idle name each rerun; the hook does the switching.

# Compact tab labels; anything not listed falls back to truncation.
SHORT_TOOL_NAMES = {
    "Brightness (WF)": "Brightness WF",
    "Brightness (Conf)": "Brightness Conf",
    "Movie Brightness": "Movie Bright",
    "Saturation Series": "Sat Series",
    "Confocal Visualization": "Confocal Viz",
    "MSR Analysis": "MSR",
    "Dye Colocalization": "Coloc",
    "Process Spectra": "Proc Spectra",
    "Spectra Saturation Series": "Spec Sat",
    "Monomer Estimation": "Monomers",
    "Shelling Injection Table": "Shelling",
    "Reaction Planner": "Rxn Planner",
    "TEM Size Analysis": "TEM Size",
    "FFT Analysis": "FFT",
}


def _short_tab_name(label: str, maxlen: int = 18) -> str:
    name = SHORT_TOOL_NAMES.get(label, label)
    if len(name) > maxlen:
        name = name[: maxlen - 1].rstrip() + "..."
    return name


def _sync_tab_title(idle_name: str):
    """Install (once) the tab-title hook and set the current idle name.

    Uses a single 0-px iframe per rerun. The hook watches Streamlit's running
    indicator (``stStatusWidget``) to show "Analyzing...", a file-input change
    to show "Uploading files...", and otherwise the idle tool name.
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
                    doc.title = 'Analyzing...';
                } else if (!(doc.title || '').startsWith('Uploading')) {
                    doc.title = doc.__psyfitIdle;
                }
            }
            if (!doc.__psyfitTitleHook) {
                doc.__psyfitTitleHook = true;
                doc.addEventListener('change', function (e) {
                    const t = e.target;
                    if (t && t.tagName === 'INPUT' && t.type === 'file' && t.files && t.files.length) {
                        doc.title = 'Uploading files...';
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
#
# One entry per tool. ``group`` organizes the Overview list only; navigation is
# a single flat list, so there is nothing to drill into. ``blurb`` should say
# what the tool produces and how it gets there. ``formats`` is what its
# uploaders accept, and ``local`` flags anything that will not work (or will not
# work well) on the shared cloud deployment.
# ---------------------------------------------------------------------------
TOOLS = {
    "Batch Convert": {
        "module": "tools.batch_convert",
        "entry": "run",
        "group": "Convert and export",
        "formats": ".sif",
        "blurb":
            "Converts many .sif acquisitions to images and a combined table in one pass. "
            "Splits each frame into quadrants, locates emitters with a peak finder (UCNP) "
            "or a blob detector (dye), and exports SVG, TIFF, PNG, or JPEG with optional "
            "fit overlays.",
    },
    "Process Movie": {
        "module": "tools.read_movie",
        "entry": "run",
        "group": "Convert and export",
        "formats": ".sif",
        "blurb":
            "Turns a .sif movie into a shareable MP4, MOV, or TIFF stack. Applies a "
            "colormap with optional log scaling and a fixed intensity range across frames, "
            "then renders each frame with acquisition labels and a colorbar.",
    },
    "Brightness (WF)": {
        "module": "tools.analyze_single_sif",
        "entry": "run",
        "group": "Brightness and intensity",
        "formats": ".sif",
        "blurb":
            "Measures per-particle brightness in widefield .sif images. Detects emitters, "
            "fits each point spread function to a 2D Gaussian, and reports the brightness "
            "distribution as a histogram plus a downloadable per-particle table.",
    },
    "Brightness (Conf)": {
        "module": "tools.confocal_brightness",
        "entry": "run",
        "group": "Brightness and intensity",
        "formats": ".dat, .tif, .txt, .csv",
        "blurb":
            "Measures per-particle brightness in confocal scans. Fits each point spread "
            "function and divides its amplitude by dwell time and line accumulation, so "
            "scans taken under different settings can be compared directly.",
    },
    "Movie Brightness": {
        "module": "tools.movie_brightness",
        "entry": "run",
        "group": "Brightness and intensity",
        "formats": ".sif",
        "beta": True,
        "local": "local only",
        "blurb":
            "Follows single-emitter brightness frame by frame through a .sif movie. Finds "
            "emitters in an accumulation image, refits each one in every frame, and applies "
            "Kalafut-Visscher step detection to recover single-dye brightness from "
            "photobleaching steps.",
    },
    "Saturation Series": {
        "module": "tools.SaturationSeries",
        "entry": "run",
        "group": "Brightness and intensity",
        "formats": ".sif",
        "blurb":
            "Plots brightness against excitation power density for a saturation series. "
            "Reads the excitation setting from each filename, fits the particles in every "
            "quadrant, and plots the mean per-particle brightness at each power point.",
    },
    "Confocal Visualization": {
        "module": "tools.confocal_visualizer",
        "entry": "run",
        "group": "Visualization",
        "formats": ".tif, .dat",
        "blurb":
            "Builds merged, figure-ready panels from confocal channels. Reads pixel size and "
            "imaging conditions out of the filenames, applies a per-channel colormap and "
            "contrast range, and arranges the channels on a labelled grid.",
    },
    "MSR Analysis": {
        "module": "tools.msr_analysis",
        "entry": "run",
        "group": "Visualization",
        "formats": ".msr",
        "beta": True,
        "blurb":
            "Opens Abberior Imspector .msr files, which pack many image stacks into one "
            "container. Lists and previews every stack, surfaces the saved acquisition "
            "metadata (excitation wavelength, detection window, dwell), and runs confocal "
            "brightness or a field-of-view overlay on the stacks you select.",
    },
    "Dye Colocalization": {
        "module": "tools.colocalization",
        "entry": "run",
        "group": "Spectral and colocalization",
        "formats": ".sif",
        "blurb":
            "Quantifies how often dye and UCNP signals sit on the same particle. Fits the "
            "point spread functions in each channel, pairs detections that fall within a "
            "distance cutoff, and reports the colocalized fraction alongside a "
            "single-emitter brightness fit for each channel.",
    },
    "Get Spectra": {
        "module": "tools.get_spectra",
        "entry": "run",
        "group": "Spectral and colocalization",
        "formats": ".sif, plus .pkl calibration",
        "blurb":
            "Extracts a per-particle emission spectrum from spectrally dispersed .sif "
            "images. Localizes each particle, maps its dispersed trace onto a wavelength "
            "axis using the uploaded calibration files, subtracts background, and exports a "
            "tidy CSV of every spectrum.",
    },
    "Process Spectra": {
        "module": "tools.process_spectra",
        "entry": "run",
        "group": "Spectral and colocalization",
        "formats": ".csv from Get Spectra",
        "blurb":
            "Compares the CSVs written by Get Spectra. Plots every spectrum, lets you click "
            "or box-select traces to exclude, then baseline-corrects, normalizes, averages, "
            "and fits Gaussian components to whatever survives the curation pass.",
    },
    "Spectra Saturation Series": {
        "module": "tools.spectra_saturation_series",
        "entry": "run",
        "group": "Spectral and colocalization",
        "formats": ".sif, plus .pkl calibration",
        "beta": True,
        "local": "folder mode is local only",
        "blurb":
            "Shows how emission spectra change with excitation power. Extracts spectra from "
            "every field of view in a series, pools particles by condition and laser "
            "current, and plots spectral shape, integrated-region ratios, and brightness "
            "against power density.",
    },
    "Monomer Estimation": {
        "module": "tools.monomers",
        "entry": "run",
        "group": "Quantification",
        "formats": ".sif",
        "blurb":
            "Estimates what fraction of a sample is monomers, dimers, trimers, or larger "
            "aggregates. Fits the per-particle brightness histogram as integer multiples of "
            "the single-particle brightness, and can convert particles per field of view "
            "into a solution concentration.",
    },
    "Shelling Injection Table": {
        "module": "tools.shelling_table",
        "entry": "run",
        "group": "Synthesis",
        "formats": "manual entry",
        "blurb":
            "Builds the injection schedule for nanocrystal shell growth. Converts a target "
            "shell thickness per injection into precursor volumes from the core and shell "
            "geometry, and flags any injection that adds more than 10% of the reaction volume.",
    },
    "Reaction Planner": {
        "module": "tools.reaction_planner",
        "entry": "run",
        "group": "Synthesis",
        "formats": "manual entry",
        "blurb":
            "Two synthesis calculators: Functionalization solves reagent amounts "
            "from a particle concentration and either a target stoichiometry or a final "
            "concentration; core synthesis converts mmol targets into mg weigh-outs for "
            "common precursors.",
    },
    "TEM Size Analysis": {
        "module": "tools.tem_analysis",
        "entry": "run",
        "group": "TEM",
        "formats": ".dm3, .emd",
        "beta": True,
        "local": "best run locally for more than 1-2 images",
        "blurb":
            "Measures particle size and shape from TEM micrographs. Separates touching "
            "particles with a watershed segmentation, fits each one to the selected shape "
            "model, and reports calibrated size distributions with Gaussian-mixture fits.",
    },
    "FFT Analysis": {
        "module": "tools.fft",
        "entry": "run",
        "group": "TEM",
        "formats": ".dm3, .emd",
        "beta": True,
        "blurb":
            "Estimates lattice spacing from a high-resolution TEM image. Takes the FFT, "
            "finds and indexes the reciprocal-lattice spots, converts them to d-spacings, "
            "and scores them against candidate host matrices.",
    },
    # "Plot CSVs": ("tools.plot_csv", "run", "Convert and export", "Flexible CSV plotting.", False),
}

OVERVIEW = "Overview"


def _tool_qualifiers(spec: dict, include_beta: bool = True) -> str:
    """Compact "what it eats, and any caveats" line, e.g. ``.sif · beta · local only``.

    Kept separate from the blurb so it can be shown inline on the Overview list
    and as a subcaption on the tool page itself.
    """
    parts = [spec["formats"]]
    if include_beta and spec.get("beta"):
        parts.append("beta")
    if spec.get("local"):
        parts.append(spec["local"])
    return " · ".join(parts)

# Preserve registry insertion order for the grouped Overview listing.
GROUP_ORDER = []
for _spec in TOOLS.values():
    if _spec["group"] not in GROUP_ORDER:
        GROUP_ORDER.append(_spec["group"])


# ---------------------------------------------------------------------------
# Sidebar: navigation
# ---------------------------------------------------------------------------
st.sidebar.title("PsyFit")
st.sidebar.caption("Microscopy and nanoparticle analysis toolkit")

selection = st.sidebar.radio(
    "Tool",
    [OVERVIEW] + list(TOOLS),
    index=0,
    format_func=lambda label: (
        label if label == OVERVIEW or not TOOLS[label].get("beta")
        else f"{label} (beta)"
    ),
)

# The settings block below is rendered *after* the selected tool has run, so a
# tool's own sidebar controls sit directly under the navigation rather than
# below the diagnostics panel. Its value is therefore read from session_state
# here and written by the keyed widget later in the same run.
SHOW_TRACES_KEY = "psyfit_show_traces"
show_traces = bool(st.session_state.get(SHOW_TRACES_KEY, False))


@st.cache_data(show_spinner=False)
def _environment_report() -> str:
    return (
        f"python: {platform.python_version()}\n"
        f"os: {platform.system()} {platform.release()}\n"
        f"executable: {sys.executable}\n"
        f"cwd: {os.getcwd()}"
    )


@st.cache_data(show_spinner=False)
def _package_report() -> str:
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

    return "\n".join(f"{p}: {pkg_version(p)}" for p in wanted_pkgs)


@st.cache_data(show_spinner=False)
def _availability_report(module_paths: tuple) -> str:
    return "\n".join(
        f"{label}: {'found' if importlib.util.find_spec(modpath) else 'MISSING'}"
        for label, modpath in module_paths
    )


def render_sidebar_settings():
    """Error-traceback toggle, diagnostics, and build stamp, pinned to the bottom.

    The three reports below are cached: an ``st.expander`` still evaluates its
    body when collapsed, and these otherwise re-scan installed distributions and
    every tool module on every rerun.
    """
    st.sidebar.divider()
    st.sidebar.toggle(
        "Show error tracebacks",
        key=SHOW_TRACES_KEY,
        help="Expand to see full Python tracebacks when tools error.",
    )

    with st.sidebar.expander("Diagnostics", expanded=False):
        st.markdown("**Environment**")
        st.code(_environment_report(), language="bash")

        st.markdown("**Package versions**")
        st.code(_package_report(), language="bash")

        st.markdown("**Tool availability (light check)**")
        st.code(
            _availability_report(
                tuple((label, spec["module"]) for label, spec in TOOLS.items())
            ),
            language="bash",
        )

        st.markdown("**Deep check a tool** (imports module)")
        deep_tool = st.selectbox("Pick a tool to deep-check:", list(TOOLS.keys()))
        if st.button("Run deep check"):
            modpath = TOOLS[deep_tool]["module"]
            funcname = TOOLS[deep_tool]["entry"]
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
    st.error(f"{title}: {type(err).__name__}: {err}")
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
    with st.spinner(f"Loading {label}..."):
        module, import_err = safe_import(modpath)
        if import_err:
            render_error_context(f"Failed to import {label} ({modpath})", import_err)
            return

    run_fn, getattr_err = safe_getattr(module, funcname)
    if getattr_err:
        render_error_context(f"Failed to find '{funcname}()' in {modpath}", getattr_err)
        return

    with st.spinner(f"Running {label}..."):
        try:
            return run_fn()
        except Exception as e:
            render_error_context(f"{label} crashed while running", e)
            return


def render_overview():
    st.title("PsyFit Tool Overview")
    st.caption(
        "A Streamlit toolkit for single-molecule microscopy and transmission electron microscope analysis."
    )

    st.divider()
    for group in GROUP_ORDER:
        st.markdown(f"**{group}**")
        for label, spec in TOOLS.items():
            if spec["group"] != group:
                continue
            st.markdown(
                f"- **{label}** *({_tool_qualifiers(spec)})*: {spec['blurb']}"
            )
        st.write("")

    st.divider()
    st.caption(
        "Built and maintained by Harrison Toll. Source: "
        "[github.com/htoll/psyfit](https://github.com/htoll/psyfit)"
    )


if selection == OVERVIEW:
    _sync_tab_title("PsyFit")
    render_overview()
else:
    spec = TOOLS[selection]
    modpath, funcname = spec["module"], spec["entry"]

    # Idle tab title = the (shortened) tool name; the hook flips it to
    # "Analyzing..." while running and "Uploading files..." during uploads.
    _sync_tab_title(_short_tab_name(selection))

    st.title(selection + (" (beta)" if spec.get("beta") else ""))
    st.caption(spec["blurb"])
    st.caption(_tool_qualifiers(spec, include_beta=False))

    st.divider()

    safe_run_tool(modpath, funcname, selection)

# Rendered last so the tool's own sidebar controls appear above it.
render_sidebar_settings()
