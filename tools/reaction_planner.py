# tools/reaction_planner.py
"""
Reaction Planner — two synthesis calculators in one tool.

1. Functionalization / Stoichiometry
   Given a nanoparticle concentration and a reagent, work out how much reagent
   to add either at a target stoichiometry (equivalents per particle) or at a
   target final concentration. Reports moles, mass (mg), and — if the reagent is
   added from a stock solution — the volume to pipette.

2. Core Synthesis (mmol → mg)
   Build a weigh-out sheet: pick reagents, enter the mmol you want, get the mg to
   weigh. Lanthanide acetates (·4H₂O), lanthanide chlorides (·6H₂O), and common
   fluoride/oleate/TFA/hydroxide precursors are built in. For the hydroxides
   (LiOH·H₂O, NaOH, KOH) you can instead dose from a methanol stock at a chosen
   molarity and get the volume to add.
"""
import io

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Reagent library
# ---------------------------------------------------------------------------
# Molar masses are computed from atomic weights so the hydration state is
# explicit and auditable. Lanthanide (+ Y) salts are the usual hydrates:
#   acetates  Ln(CH3COO)3 · 4H2O
#   chlorides LnCl3        · 6H2O
# These reproduce the lab's reference values (e.g. YCl3·6H2O = 303.36,
# Y(OAc)3·4H2O = 338.10, YbCl3·6H2O = 387.49, LiOH·H2O = 41.96).

ATOMIC = {
    "Y": 88.906, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.500,
    "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967,
    "Cl": 35.453, "H": 1.008, "O": 15.999, "C": 12.011, "Na": 22.990,
    "N": 14.007, "F": 18.998, "Li": 6.941, "K": 39.098,
}

# Rare-earth series (Pm omitted — radioactive, not used), Y grouped in.
LN_ORDER = ["Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
            "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]

_WATER = 2 * ATOMIC["H"] + ATOMIC["O"]                      # 18.015
_ACETATE = 2 * ATOMIC["C"] + 3 * ATOMIC["H"] + 2 * ATOMIC["O"]  # CH3COO 59.044
_ACETATE_ADD = 3 * _ACETATE + 4 * _WATER                   # 3 OAc + 4 H2O = 249.19
_CHLORIDE_ADD = 3 * ATOMIC["Cl"] + 6 * _WATER              # 3 Cl + 6 H2O = 214.45

# Unicode subscripts for pretty formulas.
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _sub(s: str) -> str:
    return s.translate(_SUB)


def _build_reagents():
    """Return {short_code: {formula, mm, category, is_hydroxide, solvent}}."""
    reagents = {}

    for sym in LN_ORDER:
        reagents[f"{sym}Ac"] = {
            "formula": f"{sym}(OAc){_sub('3')}·4H{_sub('2')}O",
            "mm": round(ATOMIC[sym] + _ACETATE_ADD, 2),
            "category": "Lanthanide acetate (·4H₂O)",
            "is_hydroxide": False,
        }
    for sym in LN_ORDER:
        reagents[f"{sym}Cl"] = {
            "formula": f"{sym}Cl{_sub('3')}·6H{_sub('2')}O",
            "mm": round(ATOMIC[sym] + _CHLORIDE_ADD, 2),
            "category": "Lanthanide chloride (·6H₂O)",
            "is_hydroxide": False,
        }

    # Non-lanthanide precursors. Masses verified against the lab list.
    others = [
        # short,        formula,                       molar mass, category,        hydroxide
        ("NH4F",   f"NH{_sub('4')}F",                    37.04, "Fluoride source", False),
        ("NaTFA",  f"CF{_sub('3')}COONa",               136.00, "TFA source",      False),
        ("LiTFA",  f"CF{_sub('3')}COOLi",               119.96, "TFA source",      False),
        ("Na Oleate", f"C{_sub('18')}H{_sub('33')}O{_sub('2')}Na", 304.44, "Oleate", False),
        ("Li Oleate", f"C{_sub('18')}H{_sub('33')}O{_sub('2')}Li", 288.40, "Oleate", False),
        ("LiOH",   f"LiOH·H{_sub('2')}O",                41.96, "Hydroxide",       True),
        ("NaOH",   "NaOH",                                40.00, "Hydroxide",       True),
        ("KOH",    "KOH",                                 56.11, "Hydroxide",       True),
        # Functionalization / bioconjugation
        ("EDC·HCl", f"C{_sub('8')}H{_sub('17')}N{_sub('3')}·HCl", 191.70, "Coupling reagent", False),
    ]
    for short, formula, mm, cat, is_oh in others:
        reagents[short] = {
            "formula": formula, "mm": round(mm, 2),
            "category": cat, "is_hydroxide": is_oh,
        }
    return reagents


REAGENTS = _build_reagents()
REAGENT_CODES = list(REAGENTS.keys())
HYDROXIDES = [c for c, v in REAGENTS.items() if v["is_hydroxide"]]

# --- Unit tables -----------------------------------------------------------
CONC_UNITS = {"M": 1.0, "mM": 1e-3, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12,
              "fM": 1e-15}
VOL_UNITS = {"L": 1.0, "mL": 1e-3, "µL": 1e-6}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_mass(grams: float) -> str:
    """Human-readable mass with an appropriate prefix (g / mg / µg / ng)."""
    if not np.isfinite(grams) or grams <= 0:
        return "0 g"
    for scale, unit in [(1.0, "g"), (1e-3, "mg"), (1e-6, "µg"), (1e-9, "ng")]:
        if grams >= scale:
            return f"{grams / scale:.4g} {unit}"
    return f"{grams / 1e-12:.4g} pg"


def _fmt_vol(liters: float) -> str:
    """Human-readable volume with an appropriate prefix (L / mL / µL / nL)."""
    if not np.isfinite(liters) or liters <= 0:
        return "0 L"
    for scale, unit in [(1.0, "L"), (1e-3, "mL"), (1e-6, "µL"), (1e-9, "nL")]:
        if liters >= scale:
            return f"{liters / scale:.4g} {unit}"
    return f"{liters / 1e-12:.4g} pL"


def _fmt_mol(mol: float) -> str:
    """Human-readable amount of substance (mol / mmol / µmol / nmol)."""
    if not np.isfinite(mol) or mol <= 0:
        return "0 mol"
    for scale, unit in [(1.0, "mol"), (1e-3, "mmol"), (1e-6, "µmol"),
                        (1e-9, "nmol"), (1e-12, "pmol")]:
        if mol >= scale:
            return f"{mol / scale:.4g} {unit}"
    return f"{mol / 1e-15:.4g} fmol"


# ---------------------------------------------------------------------------
# PEG molecular-weight calculator
# ---------------------------------------------------------------------------
# Model: a PEG diol H-(OCH2CH2)n-OH (= n·44.05 + H2O) whose two ends are each
# functionalized. Each end-group value below is the mass ADDED relative to a
# plain hydroxyl (–OH) terminus (so hydroxyl = 0). Simple caps are exact;
# click-chemistry / HaloTag handles are approximate and vendor/linker-dependent.
PEG_REPEAT = 2 * ATOMIC["C"] + 4 * ATOMIC["H"] + ATOMIC["O"]   # (C2H4O) 44.0526
PEG_DIOL_BASE = 2 * ATOMIC["H"] + ATOMIC["O"]                  # H2O 18.015

PEG_END_GROUPS = {
    # name:              (Δmass vs –OH end,  exact?)
    "Hydroxyl (–OH)":    (0.0,      True),
    "Methyl (–OCH₃)":    (14.027,   True),   # add CH2 (mPEG cap)
    "Amine (–NH₂)":      (-0.984,   True),   # –OH → –NH2
    "DBCO":              (276.29,   False),  # dibenzocyclooctyne amide handle
    "Tetrazine":         (252.23,   False),  # methyltetrazine handle
    "HaloTag ligand":    (199.72,   False),  # chloroalkane (HTL) handle
}
PEG_END_NAMES = list(PEG_END_GROUPS)


def _peg_mw(n_units: int, end_a: str, end_b: str) -> float:
    return (PEG_DIOL_BASE + n_units * PEG_REPEAT
            + PEG_END_GROUPS[end_a][0] + PEG_END_GROUPS[end_b][0])


def _peg_calculator_ui(key_prefix: str) -> float:
    """Render the PEG-builder inputs and return the estimated MW (g/mol)."""
    n_units = st.number_input(
        "Number of PEG units (n)", min_value=1, value=8, step=1,
        help="Ethylene-glycol repeat units, (CH₂CH₂O)ₙ, 44.05 g/mol each. "
             "For polydisperse PEG this is the average n.",
        key=f"{key_prefix}_peg_n",
    )
    pc1, pc2 = st.columns(2)
    with pc1:
        end_a = st.selectbox("End group A", PEG_END_NAMES, index=2,
                             key=f"{key_prefix}_peg_enda")
    with pc2:
        end_b = st.selectbox("End group B", PEG_END_NAMES, index=0,
                             key=f"{key_prefix}_peg_endb")
    mw = _peg_mw(int(n_units), end_a, end_b)
    st.number_input("Estimated PEG MW (g/mol)", value=float(round(mw, 2)),
                    format="%.2f", disabled=True, key=f"{key_prefix}_peg_mw")
    if not (PEG_END_GROUPS[end_a][1] and PEG_END_GROUPS[end_b][1]):
        st.caption(
            "⚠️ DBCO / tetrazine / HaloTag handle masses are approximate and "
            "vendor/linker-dependent — override MW manually if you need it exact."
        )
    return mw


# ---------------------------------------------------------------------------
# Tab 1: Functionalization / stoichiometry
# ---------------------------------------------------------------------------
def _functionalization_tab():
    st.markdown(
        "Calculate how much of a reagent to add to a nanoparticle reaction, "
        "either at a target **stoichiometry** (equivalents per particle) or at a "
        "target **final concentration**."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Reaction")
        part_val = st.number_input(
            "Particle concentration", min_value=0.0, value=100.0, step=1.0,
            format="%.4g", key="rp_part_val",
        )
        part_unit = st.selectbox(
            "Particle conc. unit", list(CONC_UNITS), index=3, key="rp_part_unit",
        )
        vol_val = st.number_input(
            "Volume of nanoparticle added", min_value=0.0, value=1.0, step=0.1,
            format="%.4g", key="rp_vol_val",
            help="Volume of the nanoparticle solution in the reaction — used as "
                 "the reaction volume for moles and final concentration.",
        )
        vol_unit = st.selectbox(
            "Volume unit", list(VOL_UNITS), index=1, key="rp_vol_unit",
        )

    with c2:
        st.subheader("Reagent")
        mw_source = st.radio(
            "Molecular weight from", ["Reagent list", "PEG calculator",
                                      "Enter manually"],
            horizontal=True, key="rp_mw_source",
        )
        if mw_source == "Reagent list":
            reagent_code = st.selectbox(
                "Reagent", REAGENT_CODES, key="rp_func_reagent",
                format_func=lambda c: f"{c} — {REAGENTS[c]['formula']}",
            )
            mw = REAGENTS[reagent_code]["mm"]
            st.number_input(
                "Molecular weight (g/mol)", value=float(mw), format="%.4f",
                disabled=True, key="rp_mw_ro",
            )
        elif mw_source == "PEG calculator":
            mw = _peg_calculator_ui("rp_func")
        else:
            mw = st.number_input(
                "Molecular weight (g/mol)", min_value=0.0, value=100.0,
                step=1.0, format="%.4f", key="rp_mw_manual",
            )

        mode = st.radio(
            "Target by", ["Stoichiometry (equiv per particle)",
                          "Final concentration"], key="rp_mode",
        )
        if mode.startswith("Stoichiometry"):
            equiv = st.number_input(
                "Equivalents per particle", min_value=0.0, value=100.0,
                step=1.0, format="%.4g", key="rp_equiv",
            )
            target_conc_M = None
        else:
            fc_val = st.number_input(
                "Target final concentration", min_value=0.0, value=10.0,
                step=1.0, format="%.4g", key="rp_fc_val",
            )
            fc_unit = st.selectbox(
                "Final conc. unit", list(CONC_UNITS), index=2, key="rp_fc_unit",
            )
            target_conc_M = fc_val * CONC_UNITS[fc_unit]
            equiv = None

    st.divider()

    # Reagents are usually dosed from a stock solution, so that is the default.
    add_as = st.radio(
        "Add reagent as", ["Stock solution", "Solid"], horizontal=True,
        key="rp_add_as",
    )
    stock_conc_M = None
    if add_as == "Stock solution":
        sc1, sc2 = st.columns(2)
        with sc1:
            stock_val = st.number_input(
                "Stock concentration", min_value=0.0, value=10.0, step=1.0,
                format="%.4g", key="rp_stock_val",
            )
        with sc2:
            stock_unit = st.selectbox(
                "Stock conc. unit", list(CONC_UNITS), index=0,
                key="rp_stock_unit",
            )
        stock_conc_M = stock_val * CONC_UNITS[stock_unit]

    # --- Compute --------------------------------------------------------
    part_conc_M = part_val * CONC_UNITS[part_unit]
    volume_L = vol_val * VOL_UNITS[vol_unit]
    n_particles = part_conc_M * volume_L                      # mol

    if volume_L <= 0 or part_conc_M <= 0:
        st.info("Enter a positive particle concentration and nanoparticle volume.")
        return
    if mw <= 0:
        st.warning("Enter a positive molecular weight.")
        return

    if equiv is not None:
        n_reagent = equiv * n_particles                       # mol
        final_conc_M = part_conc_M * equiv
    else:
        n_reagent = target_conc_M * volume_L                  # mol
        final_conc_M = target_conc_M
        equiv = (n_reagent / n_particles) if n_particles > 0 else float("nan")

    mass_g = n_reagent * mw                                    # g

    st.subheader("Result")
    if stock_conc_M:
        vol_stock_L = n_reagent / stock_conc_M
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volume of stock to add", _fmt_vol(vol_stock_L))
        m2.metric("Reagent mass", _fmt_mass(mass_g))
        m3.metric("Amount", _fmt_mol(n_reagent))
        m4.metric("Equivalents / particle", f"{equiv:.4g}")
        if vol_stock_L > volume_L:
            st.warning(
                "The stock volume to add exceeds the nanoparticle volume — "
                "use a more concentrated stock."
            )
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Reagent to weigh out", _fmt_mass(mass_g))
        m2.metric("Amount", _fmt_mol(n_reagent))
        m3.metric("Equivalents / particle", f"{equiv:.4g}")

    with st.expander("Show calculation"):
        lines = [
            f"particle conc      = {part_val:g} {part_unit} = {part_conc_M:.4g} M",
            f"nanoparticle vol   = {vol_val:g} {vol_unit} = {volume_L:.4g} L",
            f"n(particles)       = conc × volume = {_fmt_mol(n_particles)}",
            "",
        ]
        if st.session_state.get("rp_mode", "").startswith("Stoichiometry"):
            lines += [
                f"equivalents        = {equiv:g} per particle",
                f"n(reagent)         = equiv × n(particles) = {_fmt_mol(n_reagent)}",
                f"final conc         = equiv × particle conc = "
                f"{_fmt_conc(final_conc_M)}",
            ]
        else:
            lines += [
                f"target final conc  = {_fmt_conc(final_conc_M)}",
                f"n(reagent)         = conc × volume = {_fmt_mol(n_reagent)}",
                f"equivalents        = n(reagent) / n(particles) = {equiv:.4g}",
            ]
        lines += [
            "",
            f"molecular weight   = {mw:.4f} g/mol",
            f"mass to add        = n(reagent) × MW = {_fmt_mass(mass_g)}",
        ]
        if stock_conc_M:
            lines += [
                "",
                f"stock conc         = {stock_conc_M:.4g} M",
                f"volume of stock    = n(reagent) / stock conc = "
                f"{_fmt_vol(n_reagent / stock_conc_M)}",
            ]
        st.code("\n".join(lines), language="text")


def _fmt_conc(molar: float) -> str:
    if not np.isfinite(molar) or molar <= 0:
        return "0 M"
    for unit, scale in CONC_UNITS.items():
        if molar >= scale:
            return f"{molar / scale:.4g} {unit}"
    return f"{molar / 1e-12:.4g} pM"


# Lanthanide salt forms -> reagent-code suffix.
LN_SALT_FORMS = {
    "Acetate (·4H₂O)": "Ac",
    "Chloride (·6H₂O)": "Cl",
}


def _lanthanide_mixer(scale: float):
    """Total-Ln + per-lanthanide mol% mixer → mass of each salt to weigh."""
    st.subheader("Lanthanide precursors")
    st.markdown(
        "Set the **total lanthanide** amount and the **mol %** of each lanthanide, "
        "pick the salt form, and get the mass of each to weigh."
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        total_ln = st.number_input(
            "Total Ln amount (mmol)", min_value=0.0, value=2.0, step=0.1,
            format="%.4g", key="rp_ln_total",
            help="Total moles of all lanthanides combined (before the recipe "
                 "scale factor).",
        )
    with mc2:
        salt_form = st.selectbox(
            "Salt form", list(LN_SALT_FORMS), key="rp_ln_saltform",
        )
    suffix = LN_SALT_FORMS[salt_form]

    default = pd.DataFrame({
        "Lanthanide": ["Y", "Yb", "Er"],
        "mol %": [78.0, 20.0, 2.0],
    })
    comp = st.data_editor(
        default,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Lanthanide": st.column_config.SelectboxColumn(
                "Lanthanide", options=LN_ORDER, required=False,
            ),
            "mol %": st.column_config.NumberColumn(
                "mol %", min_value=0.0, max_value=100.0, format="%.4g",
            ),
        },
        key="rp_ln_editor",
    )

    total_ln_scaled = total_ln * scale

    rows = []
    pct_sum = 0.0
    for _, r in comp.iterrows():
        sym = r.get("Lanthanide")
        pct = r.get("mol %")
        if sym not in ATOMIC or pd.isna(pct) or float(pct) <= 0:
            continue
        pct = float(pct)
        pct_sum += pct
        code = f"{sym}{suffix}"
        reagent = REAGENTS.get(code)
        if reagent is None:
            continue
        mmol = total_ln_scaled * pct / 100.0
        rows.append({
            "Lanthanide": sym,
            "Reagent": code,
            "Formula": reagent["formula"],
            "mol %": pct,
            "Amount (mmol)": round(mmol, 4),
            "Weigh out (mg)": round(mmol * reagent["mm"], 2),
        })

    if not rows:
        st.info("Add at least one lanthanide with a positive mol %.")
        return

    if abs(pct_sum - 100.0) > 0.5:
        st.warning(
            f"Percentages sum to {pct_sum:g}%, not 100%. Amounts use each mol % "
            "as a fraction of the total Ln (i.e. taken literally, not normalized)."
        )

    out = pd.DataFrame(rows)
    st.dataframe(
        out.style.format({
            "mol %": "{:.4g}",
            "Amount (mmol)": "{:.4g}",
            "Weigh out (mg)": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Total lanthanide: **{out['Amount (mmol)'].sum():.4g} mmol** "
        f"({salt_form}) · total mass **{out['Weigh out (mg)'].sum():.2f} mg**"
        + (f" · recipe scale ×{scale:g}" if scale != 1.0 else "")
    )

    st.download_button(
        "Download lanthanide sheet (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="lanthanide_precursors.csv", mime="text/csv",
        key="rp_ln_csv",
    )


# ---------------------------------------------------------------------------
# Tab 2: Core synthesis (mmol -> mg)
# ---------------------------------------------------------------------------
def _core_synthesis_tab():
    scale = st.number_input(
        "Recipe scale factor", min_value=0.0, value=1.0, step=0.1,
        format="%.3g", help="Multiplies every target amount below (e.g. 2 = double batch).",
        key="rp_core_scale",
    )

    _lanthanide_mixer(scale)

    st.divider()
    st.subheader("Other solids")
    st.markdown(
        "Enter the **mmol** of each reagent you want; the sheet returns the "
        "**mg to weigh out**. Use the ➕ at the bottom of the table to add rows."
    )

    default = pd.DataFrame({
        "Reagent": ["NaOH", "NH4F"],
        "Target (mmol)": [5.0, 8.0],
    })

    edited = st.data_editor(
        default,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Reagent": st.column_config.SelectboxColumn(
                "Reagent", options=REAGENT_CODES, required=False,
            ),
            "Target (mmol)": st.column_config.NumberColumn(
                "Target (mmol)", min_value=0.0, format="%.4g",
            ),
        },
        key="rp_core_editor",
    )

    rows = []
    for _, r in edited.iterrows():
        code = r.get("Reagent")
        mmol = r.get("Target (mmol)")
        if code not in REAGENTS or pd.isna(mmol):
            continue
        mmol_scaled = float(mmol) * scale
        mm = REAGENTS[code]["mm"]
        rows.append({
            "Reagent": code,
            "Formula": REAGENTS[code]["formula"],
            "Molar mass (g/mol)": mm,
            "Target (mmol)": round(mmol_scaled, 4),
            "Weigh out (mg)": round(mmol_scaled * mm, 2),
        })

    if not rows:
        st.info("Pick a reagent and enter a target amount to see the weigh-out.")
    else:
        out = pd.DataFrame(rows)
        st.subheader("Weigh-out sheet")
        st.dataframe(
            out.style.format({
                "Molar mass (g/mol)": "{:.2f}",
                "Target (mmol)": "{:.4g}",
                "Weigh out (mg)": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Total solid mass: **{out['Weigh out (mg)'].sum():.2f} mg**")

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download weigh-out sheet (CSV)", data=csv,
            file_name="reaction_weighout.csv", mime="text/csv",
            key="rp_core_csv",
        )

    # --- Hydroxide methanol-stock dosing -------------------------------
    st.divider()
  

    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        oh_code = st.selectbox("Hydroxide", HYDROXIDES, key="rp_oh_code")
    with hc2:
        oh_mmol = st.number_input(
            "Target amount (mmol)", min_value=0.0, value=5.0, step=0.1,
            format="%.4g", key="rp_oh_mmol",
        )
    with hc3:
        oh_conc_M = st.number_input(
            "Stock conc. in methanol (M)", min_value=0.0, value=5.0, step=0.05,
            format="%.4g", key="rp_oh_conc",
        )

    oh_mmol_scaled = oh_mmol * scale
    oh_mm = REAGENTS[oh_code]["mm"]
    if oh_conc_M > 0 and oh_mmol_scaled > 0:
        vol_L = (oh_mmol_scaled * 1e-3) / oh_conc_M            # mol / (mol/L)
        r1, r2 = st.columns(2)
        r1.metric(
            f"Volume of {oh_code} stock to add", _fmt_vol(vol_L),
            help=f"{_fmt_mol(oh_mmol_scaled * 1e-3)} at {oh_conc_M:g} M",
        )
        # To prepare the stock: mg of solid per mL of methanol at this molarity.
        mg_per_mL = oh_conc_M * oh_mm
        r2.metric(
            "Stock prep", f"{mg_per_mL:.1f} mg / mL MeOH",
            help=f"Dissolve {mg_per_mL:.1f} mg {oh_code} ({REAGENTS[oh_code]['formula']}) "
                 f"per mL methanol for a {oh_conc_M:g} M stock.",
        )
        if scale != 1.0:
            st.caption(f"Amount scaled by recipe factor ×{scale:g} → "
                       f"{oh_mmol_scaled:.4g} mmol.")
    else:
        st.info("Enter a positive target amount and stock concentration.")

    # --- Reference table -----------------------------------------------
    with st.expander("Reagent molar-mass reference"):
        st.caption(
            "Lanthanide acetates are the tetrahydrate (·4H₂O); chlorides are the "
            "hexahydrate (·6H₂O); LiOH is the monohydrate (·H₂O). Masses computed "
            "from standard atomic weights."
        )
        ref = pd.DataFrame([
            {"Reagent": c, "Formula": v["formula"],
             "Molar mass (g/mol)": v["mm"], "Category": v["category"]}
            for c, v in REAGENTS.items()
        ])
        st.dataframe(
            ref.style.format({"Molar mass (g/mol)": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "Download reference table (CSV)",
            data=ref.to_csv(index=False).encode("utf-8"),
            file_name="reagent_molar_masses.csv", mime="text/csv",
            key="rp_ref_csv",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run():
    # Note: st.set_page_config is owned by app.py (called once, first); not here.
    tab_func, tab_core = st.tabs(["Functionalization", "Core synthesis"])
    with tab_func:
        _functionalization_tab()
    with tab_core:
        _core_synthesis_tab()


if __name__ == "__main__":
    run()
