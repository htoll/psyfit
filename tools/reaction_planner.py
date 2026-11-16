# tools/reaction_planner.py
"""
Reaction Planner: how much of each compound to add to a reaction.

One table, one row per compound. For each row you give
  * how it is added: weighed out as a solid, or pipetted from a stock solution
    of known concentration,
  * its molecular weight (needed for solids and mg/mL stocks; filled in from the
    built-in reagent list when the name matches one),
  * a target, either a final concentration in the reaction or an amount (mol).

The reaction has a fixed final volume. Concentration targets are relative to that
volume, stock additions count toward it, and the tool reports how much solvent to
top up with after all stocks are in. Solids are assumed to add negligible volume.
"""
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Reagent library (MW lookup by name)
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

# Rare-earth series (Pm omitted: radioactive, not used), Y grouped in.
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
    """Return {short_code: {formula, mm, category}}."""
    reagents = {}

    for sym in LN_ORDER:
        reagents[f"{sym}Ac"] = {
            "formula": f"{sym}(OAc){_sub('3')}·4H{_sub('2')}O",
            "mm": round(ATOMIC[sym] + _ACETATE_ADD, 2),
            "category": "Lanthanide acetate (·4H₂O)",
        }
    for sym in LN_ORDER:
        reagents[f"{sym}Cl"] = {
            "formula": f"{sym}Cl{_sub('3')}·6H{_sub('2')}O",
            "mm": round(ATOMIC[sym] + _CHLORIDE_ADD, 2),
            "category": "Lanthanide chloride (·6H₂O)",
        }

    # Non-lanthanide precursors. Masses verified against the lab list.
    others = [
        # short,        formula,                       molar mass, category
        ("NH4F",   f"NH{_sub('4')}F",                    37.04, "Fluoride source"),
        ("NaTFA",  f"CF{_sub('3')}COONa",               136.00, "TFA source"),
        ("LiTFA",  f"CF{_sub('3')}COOLi",               119.96, "TFA source"),
        ("Na Oleate", f"C{_sub('18')}H{_sub('33')}O{_sub('2')}Na", 304.44, "Oleate"),
        ("Li Oleate", f"C{_sub('18')}H{_sub('33')}O{_sub('2')}Li", 288.40, "Oleate"),
        ("LiOH",   f"LiOH·H{_sub('2')}O",                41.96, "Hydroxide"),
        ("NaOH",   "NaOH",                                40.00, "Hydroxide"),
        ("KOH",    "KOH",                                 56.11, "Hydroxide"),
        # Functionalization / bioconjugation
        ("EDC·HCl", f"C{_sub('8')}H{_sub('17')}N{_sub('3')}·HCl", 191.70, "Coupling reagent"),
    ]
    for short, formula, mm, cat in others:
        reagents[short] = {"formula": formula, "mm": round(mm, 2), "category": cat}
    return reagents


REAGENTS = _build_reagents()


def _norm_name(name: str) -> str:
    """Lookup key: case-, space- and punctuation-insensitive ("edc hcl" → "edchcl")."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# Each reagent is findable by its short code or its formula; EDC also by "EDC".
_LOOKUP = {}
for _code, _r in REAGENTS.items():
    _LOOKUP[_norm_name(_code)] = _code
    _LOOKUP[_norm_name(_r["formula"].translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")))] = _code
_LOOKUP[_norm_name("EDC")] = "EDC·HCl"


def lookup_mw(name: str):
    """(code, MW) for a compound name that matches the built-in list, else (None, nan)."""
    code = _LOOKUP.get(_norm_name(name))
    return (code, REAGENTS[code]["mm"]) if code else (None, float("nan"))


# --- Unit tables -----------------------------------------------------------
CONC_UNITS = {"M": 1.0, "mM": 1e-3, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12}
AMOUNT_UNITS = {"mol": 1.0, "mmol": 1e-3, "µmol": 1e-6, "nmol": 1e-9, "pmol": 1e-12}
VOL_UNITS = {"L": 1.0, "mL": 1e-3, "µL": 1e-6}
# A stock can be given in molar units or, for things usually made up by weight
# (polymers, proteins), in mg/mL, which needs the MW to convert.
STOCK_UNITS = list(CONC_UNITS) + ["mg/mL"]
TARGET_UNITS = list(CONC_UNITS) + list(AMOUNT_UNITS)

SOLID, STOCK = "Solid", "Stock solution"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_scaled(value: float, scales) -> str:
    if not np.isfinite(value) or value <= 0:
        return ""
    for scale, unit in scales:
        if value >= scale:
            return f"{value / scale:.4g} {unit}"
    scale, unit = scales[-1]
    return f"{value / scale:.4g} {unit}"


def _fmt_mass(grams: float) -> str:
    return _fmt_scaled(grams, [(1.0, "g"), (1e-3, "mg"), (1e-6, "µg"), (1e-9, "ng")])


def _fmt_vol(liters: float) -> str:
    return _fmt_scaled(liters, [(1.0, "L"), (1e-3, "mL"), (1e-6, "µL"), (1e-9, "nL")])


def _fmt_mol(mol: float) -> str:
    return _fmt_scaled(mol, [(1.0, "mol"), (1e-3, "mmol"), (1e-6, "µmol"),
                             (1e-9, "nmol"), (1e-12, "pmol")])


def _fmt_conc(molar: float) -> str:
    return _fmt_scaled(molar, [(s, u) for u, s in CONC_UNITS.items()])


# ---------------------------------------------------------------------------
# PEG molecular-weight calculator (MW lookup aid)
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


def _peg_calculator_ui() -> None:
    n_units = st.number_input(
        "Number of PEG units (n)", min_value=1, value=8, step=1,
        help="Ethylene-glycol repeat units, (CH₂CH₂O)ₙ, 44.05 g/mol each. "
             "For polydisperse PEG this is the average n.",
        key="rp_peg_n",
    )
    pc1, pc2 = st.columns(2)
    with pc1:
        end_a = st.selectbox("End group A", PEG_END_NAMES, index=2, key="rp_peg_enda")
    with pc2:
        end_b = st.selectbox("End group B", PEG_END_NAMES, index=0, key="rp_peg_endb")
    mw = _peg_mw(int(n_units), end_a, end_b)
    st.markdown(f"**PEG MW ≈ {mw:.2f} g/mol**. Copy this into the MW column.")
    if not (PEG_END_GROUPS[end_a][1] and PEG_END_GROUPS[end_b][1]):
        st.caption("DBCO / tetrazine / HaloTag handle masses are approximate and "
                   "vendor/linker-dependent.")


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------
def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def compute_row(name, added_as, mw_in, stock_val, stock_unit, target_val, target_unit,
                volume_L: float) -> dict:
    """Amounts for one compound. Returns a dict of raw SI values plus an ``issue`` string
    (empty when the row is complete). Amounts are in mol, volumes in L, masses in g."""
    out = {"mol": float("nan"), "conc_M": float("nan"), "mass_g": float("nan"),
           "stock_L": float("nan"), "mw": float("nan"), "mw_source": "", "issue": ""}

    mw = _num(mw_in)
    if np.isfinite(mw) and mw > 0:
        out["mw_source"] = "entered"
    else:
        code, mw = lookup_mw(name)
        if code:
            out["mw_source"] = f"library ({code}, {REAGENTS[code]['formula']})"
    out["mw"] = mw
    has_mw = np.isfinite(mw) and mw > 0

    target = _num(target_val)
    if not (np.isfinite(target) and target > 0) or target_unit not in TARGET_UNITS:
        out["issue"] = "enter a target"
        return out
    if target_unit in CONC_UNITS:
        out["conc_M"] = target * CONC_UNITS[target_unit]
        out["mol"] = out["conc_M"] * volume_L
    else:
        out["mol"] = target * AMOUNT_UNITS[target_unit]
        out["conc_M"] = out["mol"] / volume_L
    if has_mw:
        out["mass_g"] = out["mol"] * mw

    if added_as == STOCK:
        sv = _num(stock_val)
        if not (np.isfinite(sv) and sv > 0) or stock_unit not in STOCK_UNITS:
            out["issue"] = "enter the stock concentration"
            return out
        if stock_unit == "mg/mL":
            if not has_mw:
                out["issue"] = "mg/mL stock needs an MW"
                return out
            stock_M = sv / mw                      # mg/mL = g/L → mol/L
        else:
            stock_M = sv * CONC_UNITS[stock_unit]
        out["stock_L"] = out["mol"] / stock_M
        if stock_M <= out["conc_M"]:
            out["issue"] = "stock is not more concentrated than the target"
    elif not has_mw:
        out["issue"] = "enter an MW (name not in the reagent list)"
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_DEFAULT_ROWS = pd.DataFrame({
    "Compound": ["NH4F", "EDC", "Ligand"],
    "Added as": [SOLID, STOCK, STOCK],
    "MW (g/mol)": [None, None, 500.0],
    "Stock conc": [None, 100.0, 10.0],
    "Stock unit": ["mM", "mM", "mg/mL"],
    "Target": [8.0, 5.0, 0.5],
    "Target unit": ["mmol", "mM", "µmol"],
})


def run():
    # Note: st.set_page_config is owned by app.py (called once, first); not here.
    st.markdown(
        "One row per compound. Choose whether it is **weighed out** or added from a "
        "**stock solution**, and give a target as either a **final concentration** "
        "(M, mM, …) or an **amount** (mol, mmol, …). MW is filled in from the built-in "
        "list when the name matches (e.g. NH4F, YbCl, LiOH, EDC); otherwise enter it."
    )

    vc1, vc2, _ = st.columns([1, 1, 2])
    with vc1:
        vol_val = st.number_input("Final reaction volume", min_value=0.0, value=1.0,
                                  step=0.1, format="%.4g", key="rp_vol_val")
    with vc2:
        vol_unit = st.selectbox("Volume unit", list(VOL_UNITS), index=1, key="rp_vol_unit")
    volume_L = vol_val * VOL_UNITS[vol_unit]

    table = st.data_editor(
        _DEFAULT_ROWS, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "Compound": st.column_config.TextColumn("Compound", required=False),
            "Added as": st.column_config.SelectboxColumn(
                "Added as", options=[SOLID, STOCK], default=SOLID),
            "MW (g/mol)": st.column_config.NumberColumn(
                "MW (g/mol)", min_value=0.0, format="%.4g",
                help="Leave blank to use the built-in list. Needed for solids and mg/mL stocks."),
            "Stock conc": st.column_config.NumberColumn(
                "Stock conc", min_value=0.0, format="%.4g",
                help="Only used when the compound is added from a stock solution."),
            "Stock unit": st.column_config.SelectboxColumn(
                "Stock unit", options=STOCK_UNITS, default="mM"),
            "Target": st.column_config.NumberColumn("Target", min_value=0.0, format="%.4g"),
            "Target unit": st.column_config.SelectboxColumn(
                "Target unit", options=TARGET_UNITS, default="mM",
                help="A concentration unit means final concentration in the reaction; "
                     "an amount unit means total moles added."),
        },
        key="rp_table",
    )

    if volume_L <= 0:
        st.info("Enter a positive final reaction volume.")
        return

    rows, csv_rows, total_stock_L = [], [], 0.0
    for _, r in table.iterrows():
        name = r.get("Compound")
        if name is None or (isinstance(name, float) and np.isnan(name)) or not str(name).strip():
            continue
        name = str(name).strip()
        added_as = r.get("Added as") or SOLID
        res = compute_row(name, added_as, r.get("MW (g/mol)"), r.get("Stock conc"),
                          r.get("Stock unit"), r.get("Target"), r.get("Target unit"), volume_L)
        if np.isfinite(res["stock_L"]):
            total_stock_L += res["stock_L"]

        if added_as == STOCK:
            add = f"{_fmt_vol(res['stock_L'])} of stock" if np.isfinite(res["stock_L"]) else ""
        else:
            add = f"weigh {_fmt_mass(res['mass_g'])}" if np.isfinite(res["mass_g"]) else ""
        rows.append({
            "Compound": name,
            "Add": add,
            "Amount": _fmt_mol(res["mol"]),
            "Final conc": _fmt_conc(res["conc_M"]),
            "Mass": _fmt_mass(res["mass_g"]),
            "MW used": (f"{res['mw']:.4g} ({res['mw_source']})"
                        if np.isfinite(res["mw"]) and res["mw"] > 0 else ""),
            "Check": res["issue"],
        })
        csv_rows.append({
            "compound": name, "added_as": added_as,
            "mw_g_per_mol": res["mw"], "mw_source": res["mw_source"],
            "stock_conc": _num(r.get("Stock conc")) if added_as == STOCK else float("nan"),
            "stock_unit": r.get("Stock unit") if added_as == STOCK else "",
            "target": _num(r.get("Target")), "target_unit": r.get("Target unit"),
            "amount_mol": res["mol"], "final_conc_M": res["conc_M"],
            "mass_mg": res["mass_g"] * 1e3, "stock_volume_uL": res["stock_L"] * 1e6,
            "issue": res["issue"],
        })

    if not rows:
        st.info("Add a compound to see amounts.")
        return

    st.subheader("What to add")
    out = pd.DataFrame(rows)
    if not out["Check"].astype(bool).any():
        out = out.drop(columns="Check")
    st.dataframe(out, use_container_width=True, hide_index=True)

    solvent_L = volume_L - total_stock_L
    m1, m2, m3 = st.columns(3)
    m1.metric("Final volume", _fmt_vol(volume_L))
    m2.metric("Total stock added", _fmt_vol(total_stock_L) or "0")
    m3.metric("Top up with solvent", _fmt_vol(solvent_L) if solvent_L > 0 else "none")
    if solvent_L < 0:
        st.error(f"Stock additions ({_fmt_vol(total_stock_L)}) exceed the final volume "
                 f"by {_fmt_vol(-solvent_L)}. Use more concentrated stocks or a larger volume.")
    st.caption("Solids are assumed to add no volume. Weigh solids in, add the stocks, "
               "then top up with solvent to the final volume.")

    summary = pd.DataFrame(csv_rows)
    summary["final_volume_L"] = volume_L
    summary["solvent_top_up_uL"] = solvent_L * 1e6
    st.download_button(
        "Download plan (CSV)", data=summary.to_csv(index=False).encode("utf-8"),
        file_name="reaction_plan.csv", mime="text/csv", key="rp_csv",
    )

    with st.expander("PEG MW calculator"):
        _peg_calculator_ui()

    with st.expander("Built-in reagent list"):
        st.caption("Type any of these codes in the Compound column to use its MW. "
                   "Lanthanide acetates are ·4H₂O, chlorides ·6H₂O, LiOH is ·H₂O.")
        ref = pd.DataFrame([
            {"Code": c, "Formula": v["formula"], "MW (g/mol)": v["mm"], "Category": v["category"]}
            for c, v in REAGENTS.items()
        ])
        st.dataframe(ref.style.format({"MW (g/mol)": "{:.2f}"}),
                     use_container_width=True, hide_index=True)


if __name__ == "__main__":
    run()
