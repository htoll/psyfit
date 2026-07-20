"""End-to-end smoke test: synthetic lattice image → FFT → spot detection → blind index.
Run: python tools/verify_fft_pipeline.py"""
from __future__ import annotations

import numpy as np

import crystallography as xtal
import fft


def synth_image(phase_key, uvw, nm_per_px=0.02, n=256, n_refl=8):
    """Build a real-space lattice image from a phase's zone pattern."""
    lat = xtal.Lattice.from_phase(xtal.PHASES[phase_key])
    pred = xtal.zone_pattern(lat, uvw, d_min_nm=0.15, d_max_nm=1.2, hkl_max=5)
    pred = pred[:n_refl]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    x_nm, y_nm = xx * nm_per_px, yy * nm_per_px
    img = np.zeros((n, n))
    for r in pred:
        gx, gy = r.g2d
        img += np.cos(2 * np.pi * (gx * x_nm + gy * y_nm))
    img = (img - img.min()) / (img.ptp() + 1e-9)
    return img.astype(np.float32)


def run_case(phase_key, uvw):
    nm_per_px = 0.02
    img = synth_image(phase_key, uvw, nm_per_px=nm_per_px)
    fft_mag, _ = fft.process_fft_image.__wrapped__(img)  # bypass st cache
    spots = fft.detect_spots(fft_mag, nm_per_px, img.shape,
                             sensitivity=0.05, min_dist_px=4, dc_radius=6, max_spots=24)
    measured = spots[["gx", "gy"]].to_numpy()
    sols = xtal.index_pattern(measured, tol_frac=0.06, top_n=3)
    top = sols[0] if sols else None
    truth = xtal.PHASES[phase_key]
    ok = top is not None and top.phase.key == phase_key
    got = top.phase.label if top else "—"
    print(f"  truth {truth.label:24s} {uvw} → {len(spots):2d} spots → "
          f"{got:24s} cov={top.coverage:.2f} scale={top.scale:.3f} "
          f"{'OK' if ok else 'FAIL'}")
    assert ok, f"{phase_key}: got {got}"
    assert top.coverage >= 0.5


if __name__ == "__main__":
    print("── Synthetic image → FFT → index ──")
    run_case("beta_NaYF4", (0, 0, 1))
    run_case("alpha_NaYF4", (0, 0, 1))
    run_case("alpha_NaYF4", (0, 1, 1))
    run_case("LiYF4", (0, 0, 1))
    run_case("beta_NaYbF4", (0, 0, 1))
    print("Pipeline smoke test passed.")
