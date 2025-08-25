import math
import numpy as np
import pandas as pd
import streamlit as st

def run():
  st.set_page_config(
      page_title="UCNP Shelling Injection Planner",
      page_icon="🧪",
      layout="centered",
  )
  
  st.title("🧪 UCNP Shelling Injection Planner")
  st.caption("Simple estimation of the number of injections and volumes for shelling UCNPs.")
  
  with st.form("inputs"):
      col1, col2 = st.columns(2)
      with col1:
          delta = st.number_input(
              "Δ thickness per injection (nm)",
              min_value=0.01,
              value=0.25,
              step=0.01,
              help=(
                  "Desired increase in shell thickness per injection. "
                  "Example: 8 nm core grows to 8.5 nm after a single injection → Δ = 0.5"
              ),
              format="%0.2f",
          )
          initial_radius = st.number_input(
              "Initial core radius (nm)",
              min_value=0.0,
              value=13.7/2,
              step=0.05,
              format="%0.2f",
          )
          final_radius = st.number_input(
              "Target final radius (nm)",
              min_value=0.0,
              value=19/2,
              step=0.05,
              format="%0.2f",
          )
      with col2:
          nm_per_mL = st.number_input(
              "nm³ of shell per mL YAc",
              min_value=1,
              value=200,
              step=1,
              help=(
                  "Estimated nanoparticle volume (nm³) grown per mL of YAc precursor added."
              ),
          )
          injection_time = st.number_input(
              "Delay between injections (min)",
              min_value=1,
              value=20,
              step=1,
          )
          initial_rxn_vol = st.number_input(
              "Initial reaction volume (mL)",
              min_value=0.1,
              value=10.0,
              step=0.1,
          )
  
      submitted = st.form_submit_button("Calculate")
  
  
  def compute_injection_plan(delta: float,
                              initial_radius: float,
                              final_radius: float,
                              nm_per_mL: float = 200,
                              injection_time: int = 20,
                              initial_rxn_vol: float = 10.0):
      """Return a (DataFrame, warnings) with the injection plan.
  
      Notes mirror the original script's logic:
      - Number of injections = ceil((final - initial)/Δ)
      - Estimated radius increases by Δ each step (starting from initial)
      - Injection times are cumulative (e.g., 20, 40, 60 ...)
      - YAc for step i is based on Δ-volume between radii i and i+1, scaled by nm_per_mL
      - NaTFA for step i+1 is half of previous YAc (first injection has 0)
      - Total reaction volume updates cumulatively
      - Warn if a single injection volume addition exceeds 10% of current volume
      """
      warnings = []
  
      if delta <= 0:
          raise ValueError("Δ thickness per injection must be > 0")
      if final_radius <= initial_radius:
          raise ValueError("Final radius must be greater than initial radius")
  
      num_injections = math.ceil((final_radius - initial_radius) / delta)
  
      # Arrays
      inj_numbers = np.arange(1, num_injections + 1)
      est_radius = np.array([initial_radius + i * delta for i in range(num_injections)])
      inj_times = np.array([injection_time * (i + 1) for i in range(num_injections)], dtype=float)
  
      # Volume grown per step (between radius i and i+1)
      # Only defined for the first (num_injections - 1) transitions
      volume_added = np.zeros(num_injections - 1)
      for v in range(num_injections - 1):
          r1, r2 = est_radius[v], est_radius[v + 1]
          volume_added[v] = (4.0 / 3.0) * math.pi * (r2 ** 3 - r1 ** 3)
  
      # YAc: mL added per injection (last injection has 0 mL as in original logic)
      yac_added = np.zeros(num_injections)
      for y in range(num_injections - 1):
          yac_added[y] = round(volume_added[y] / nm_per_mL, 2)
  
      # NaTFA: half of previous YAc, shifted by one (first is 0)
      tfa_added = np.zeros(num_injections)
      for t in range(num_injections - 1):
          tfa_added[t + 1] = round(yac_added[t] / 2.0, 2)
  
      # Total reaction volume & percent injected
      total_vol = np.zeros(num_injections)
      pct_injected = np.zeros(num_injections)
      prev_vol = float(initial_rxn_vol)
      for q in range(num_injections):
          this_add = yac_added[q] + tfa_added[q]
          total_vol[q] = round(prev_vol + this_add, 2)
          pct = (total_vol[q] - prev_vol) / prev_vol * 100.0 if prev_vol > 0 else 0.0
          pct_injected[q] = round(pct, 2)
          if pct > 10.0:
              warnings.append(
                  f"Warning (Injection {q+1}): >10% volume added in a single step may cause temperature fluctuations."
              )
          prev_vol = total_vol[q]
  
      # Build a tidy, per-injection table
      df = pd.DataFrame({
          "Injection": inj_numbers,
          "Time (min)": inj_times,
          "Estimated radius (nm)": np.round(est_radius, 3),
          "NaTFA (mL)": tfa_added,
          "YAc (mL)": yac_added,
          "Total Rxn Volume (mL)": total_vol,
          "% Volume Injected": pct_injected,
      })
  
      return df, warnings
  
  
  if submitted:
      try:
          df, warnings = compute_injection_plan(
              delta=delta,
              initial_radius=initial_radius,
              final_radius=final_radius,
              nm_per_mL=nm_per_mL,
              injection_time=injection_time,
              initial_rxn_vol=initial_rxn_vol,
          )
  
          st.subheader("Injection Table")
          st.dataframe(
              df,
              use_container_width=True,
              hide_index=True,
          )
  
          csv = df.to_csv(index=False).encode("utf-8")
          st.download_button(
              label="Download CSV",
              data=csv,
              file_name="ucnp_injection_plan.csv",
              mime="text/csv",
          )
  
          if warnings:
              st.warning("\n".join(warnings))
  
          with st.expander("Details & assumptions"):
              st.markdown(
                  """
                  - Number of injections = ceil((final − initial)/Δ).
                  - Radii increase by Δ per step starting at the initial radius.
                  - **YAc (mL)** is computed from the incremental shell volume divided by `nm³ per mL YAc`.
                  - **NaTFA (mL)** for injection *i* is 0 for the first and half of the previous YAc thereafter.
                  - **% Volume Injected** compares the per-step addition to the running reaction volume.
                  - This reproduces the behavior of the original script while presenting results per injection row.
                  """
              )
  
      except Exception as e:
          st.error(str(e))
  
