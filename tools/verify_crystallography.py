"""Headless checks for tools/crystallography.py — run: python tools/verify_crystallography.py"""
from __future__ import annotations

import numpy as np

from crystallography import (
    PHASES, Lattice, d_spacing_nm, is_allowed, zone_pattern,
    index_pattern, interplanar_angle_deg, candidate_zone_axes,
)


def approx(a, b, tol=0.01):
    return abs(a - b) <= tol


def check_known_dspacings():
    print("── Known d-spacings (nm) ──")
    # Hexagonal β-NaYF4: (100) d = a*sqrt(3)/2, (110) d = a/2, (002) d = c/2
    beta = PHASES["beta_NaYF4"]
    a, c = beta.a, beta.c
    exp = {
        (1, 0, 0): (a * np.sqrt(3) / 2) / 10,
        (1, 1, 0): (a / 2) / 10,
        (0, 0, 2): (c / 2) / 10,
    }
    for hkl, d in exp.items():
        got = d_spacing_nm(beta, hkl)
        ok = approx(got, d, 0.002)
        print(f"  β-NaYF4 {hkl}: {got:.4f}  expected {d:.4f}  {'OK' if ok else 'FAIL'}")
        assert ok, hkl

    # Cubic α-NaYF4: (111) d = a/sqrt(3), (200) d = a/2
    alpha = PHASES["alpha_NaYF4"]
    a = alpha.a
    for hkl, d in {(1, 1, 1): (a / np.sqrt(3)) / 10, (2, 0, 0): (a / 2) / 10}.items():
        got = d_spacing_nm(alpha, hkl)
        ok = approx(got, d, 0.002)
        print(f"  α-NaYF4 {hkl}: {got:.4f}  expected {d:.4f}  {'OK' if ok else 'FAIL'}")
        assert ok, hkl

    # Tetragonal LiYF4: (200) d = a/2, (004) d = c/4
    lyf = PHASES["LiYF4"]
    for hkl, d in {(2, 0, 0): (lyf.a / 2) / 10, (0, 0, 4): (lyf.c / 4) / 10}.items():
        got = d_spacing_nm(lyf, hkl)
        ok = approx(got, d, 0.002)
        print(f"  LiYF4   {hkl}: {got:.4f}  expected {d:.4f}  {'OK' if ok else 'FAIL'}")
        assert ok, hkl


def check_absences():
    print("── Systematic absences ──")
    # Cubic F: (100),(110) forbidden; (111),(200),(220) allowed
    a = PHASES["alpha_NaYF4"]
    assert not is_allowed(a, (1, 0, 0)) and not is_allowed(a, (1, 1, 0))
    assert is_allowed(a, (1, 1, 1)) and is_allowed(a, (2, 0, 0)) and is_allowed(a, (2, 2, 0))
    # Scheelite I4_1/a: (100) forbidden (odd sum), (002) forbidden (00l needs l=4n),
    # (004) allowed, (200) allowed, (101) allowed (1+0+1=2 even)
    t = PHASES["LiYF4"]
    assert not is_allowed(t, (1, 0, 0))
    assert not is_allowed(t, (0, 0, 2))
    assert is_allowed(t, (0, 0, 4))
    assert is_allowed(t, (2, 0, 0))
    assert is_allowed(t, (1, 0, 1))
    print("  cubic-F and I4_1/a absence rules OK")


def check_zone_angles():
    print("── Zone-pattern geometry ──")
    # β-NaYF4 down [0001]: the {10-10} spots form a hexagon → 60° between g(100) & g(010).
    lat = Lattice.from_phase(PHASES["beta_NaYF4"])
    pred = zone_pattern(lat, (0, 0, 1), d_min_nm=0.1, d_max_nm=1.0, hkl_max=4)
    by_hkl = {r.hkl: r for r in pred}
    assert (1, 0, 0) in by_hkl and (0, 1, 0) in by_hkl, "missing prism reflections in [0001]"
    g1, g2 = by_hkl[(1, 0, 0)].g2d, by_hkl[(0, 1, 0)].g2d
    ang = np.degrees(np.arccos(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))))
    print(f"  β-NaYF4 [0001] angle (100)^(010) = {ang:.1f}°  (expect 60°)")
    assert approx(ang, 60.0, 1.0)

    # Cubic [001]: (200)^(020) = 90°
    latc = Lattice.from_phase(PHASES["alpha_NaYF4"])
    predc = zone_pattern(latc, (0, 0, 1), 0.1, 1.0, hkl_max=4)
    bh = {r.hkl: r for r in predc}
    g1, g2 = bh[(2, 0, 0)].g2d, bh[(0, 2, 0)].g2d
    ang = np.degrees(np.arccos(np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))))
    print(f"  α-NaYF4 [001] angle (200)^(020) = {ang:.1f}°  (expect 90°)")
    assert approx(ang, 90.0, 1.0)


def check_blind_index():
    print("── Blind indexing of synthetic patterns ──")
    for key, uvw in [("beta_NaYF4", (0, 0, 1)),
                     ("alpha_NaYF4", (0, 1, 1)),
                     ("LiYF4", (0, 0, 1))]:
        phase = PHASES[key]
        lat = Lattice.from_phase(phase)
        pred = zone_pattern(lat, uvw, 0.1, 1.0, hkl_max=5)
        # Build a synthetic measured set: apply an arbitrary rotation + 1% scale noise.
        theta = np.radians(23.4)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rng = np.random.RandomState(0)
        spots = []
        for r in pred[:8]:
            noisy = r.g2d * (1.0 + rng.uniform(-0.01, 0.01))
            spots.append(R @ noisy)
        spots = np.array(spots)
        sols = index_pattern(spots, tol_frac=0.06, min_matched=3, top_n=3)
        assert sols, f"no solution for {key} {uvw}"
        top = sols[0]
        red = tuple(int(x) for x in top.uvw)
        want = tuple(int(x) for x in uvw)
        ok_phase = top.phase.key == key
        # Zone axis may come back as an equivalent (e.g. sign/permutation) direction;
        # accept a match on phase + high coverage as the primary success criterion.
        print(f"  truth {phase.label:26s} {want} → got {top.phase.label:26s} {red} "
              f"cov={top.coverage:.2f} scale={top.scale:.3f} {'OK' if ok_phase else 'FAIL'}")
        assert ok_phase, f"phase mismatch for {key}: got {top.phase.key}"
        assert top.coverage >= 0.6


def check_discrimination():
    print("── Phase discrimination (hexagonal vs cubic) ──")
    # A real β-NaYF4 [0001] hexagonal net must beat every cubic solution.
    lat = Lattice.from_phase(PHASES["beta_NaYF4"])
    pred = zone_pattern(lat, (0, 0, 1), 0.1, 1.0, hkl_max=5)
    spots = np.array([r.g2d for r in pred[:8]])
    sols = index_pattern(spots, tol_frac=0.05, top_n=6)
    top = sols[0]
    print(f"  best = {top.phase.label} {top.uvw} (cov {top.coverage:.2f}); "
          f"system={top.phase.system}")
    assert top.phase.system == "hexagonal", "hexagonal net misidentified as non-hexagonal"


if __name__ == "__main__":
    check_known_dspacings()
    check_absences()
    check_zone_angles()
    check_blind_index()
    check_discrimination()
    print("\nAll crystallography checks passed. Zone axes searched:",
          len(candidate_zone_axes(2)))
