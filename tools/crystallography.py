"""Crystallographic engine for indexing FFTs of HRTEM lattice images.

This module is deliberately free of Streamlit / plotting dependencies so the physics
can be unit-tested headlessly (see ``if __name__ == "__main__"`` at the bottom, or
``tools/verify_crystallography.py``).

What it does
────────────
Given a set of diffraction spots measured from the FFT of an atomic-resolution image
(each spot is a reciprocal-lattice vector **g** = 1/d, in 1/nm, relative to the DC term),
it searches every candidate phase and every low-index zone axis [uvw] for the crystal
orientation whose predicted 2-D reciprocal net best reproduces the measured spots. It
returns a ranked list of solutions, each carrying the phase, the zone axis, an (hkl)
assignment for every matched spot, and a fitted scale factor that doubles as a
calibration cross-check.

Reference data
──────────────
Lattice parameters are literature values for the six host matrices requested; every
number is editable in ``PHASES`` below and carries its source in ``Phase.reference``.
Systematic absences are applied per space group so only *observable* reflections are
predicted.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import gcd
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Phase definitions (literature lattice parameters, in ångström)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Phase:
    key: str
    label: str
    system: str          # 'cubic' | 'hexagonal' | 'tetragonal'
    a: float             # Å
    c: float             # Å (ignored for cubic)
    space_group: str
    reference: str
    color: str = "#00e5ff"

    @property
    def cell(self) -> Tuple[float, float, float, float, float, float]:
        """(a, b, c, alpha, beta, gamma) with lengths in Å and angles in degrees."""
        if self.system == "cubic":
            return (self.a, self.a, self.a, 90.0, 90.0, 90.0)
        if self.system == "hexagonal":
            return (self.a, self.a, self.c, 90.0, 90.0, 120.0)
        if self.system == "tetragonal":
            return (self.a, self.a, self.c, 90.0, 90.0, 90.0)
        raise ValueError(f"Unknown crystal system: {self.system}")


# Absence rules return True when a reflection is ALLOWED (observable).
def _allowed_F(h: int, k: int, l: int) -> bool:
    """Face-centred (F): indices all even or all odd (unmixed parity)."""
    return (h % 2 == k % 2) and (k % 2 == l % 2)


def _allowed_I41a(h: int, k: int, l: int) -> bool:
    """I4_1/a (scheelite): body-centring + 4_1 screw + a-glide conditions.

    hkl: h+k+l = 2n ; hk0: h,k = 2n ; 00l: l = 4n (others follow from these).
    """
    if (h + k + l) % 2 != 0:
        return False
    if l == 0 and (h % 2 != 0 or k % 2 != 0):
        return False
    if h == 0 and k == 0 and l % 4 != 0:
        return False
    return True


def _allowed_P(h: int, k: int, l: int) -> bool:
    """Primitive hexagonal P6̄ — no centring/systematic absences."""
    return True


_ABSENCE_RULES: Dict[str, Callable[[int, int, int], bool]] = {
    "cubic": _allowed_F,          # Fm-3m for both cubic fluorite phases here
    "hexagonal": _allowed_P,      # P6̄ (#174)
    "tetragonal": _allowed_I41a,  # I4_1/a (#88)
}


PHASES: Dict[str, Phase] = {
    "alpha_NaYF4": Phase(
        "alpha_NaYF4", "α-NaYF₄ (cubic)", "cubic", a=5.470, c=5.470,
        space_group="Fm-3m (225)", reference="JCPDS 77-2042", color="#4cc9f0"),
    "beta_NaYF4": Phase(
        "beta_NaYF4", "β-NaYF₄ (hexagonal)", "hexagonal", a=5.9688, c=3.5090,
        space_group="P6̄ (174)", reference="Krämer et al. 2004; JCPDS 16-0334", color="#f72585"),
    "alpha_NaYbF4": Phase(
        "alpha_NaYbF4", "α-NaYbF₄ (cubic)", "cubic", a=5.435, c=5.435,
        space_group="Fm-3m (225)", reference="JCPDS 77-2043", color="#4895ef"),
    "beta_NaYbF4": Phase(
        "beta_NaYbF4", "β-NaYbF₄ (hexagonal)", "hexagonal", a=5.906, c=3.464,
        space_group="P6̄ (174)", reference="JCPDS 27-1427", color="#b5179e"),
    "LiYF4": Phase(
        "LiYF4", "LiYF₄ (tetragonal)", "tetragonal", a=5.1668, c=10.735,
        space_group="I4₁/a (88)", reference="Goryunov et al. 1992", color="#ffb703"),
    "LiYbF4": Phase(
        "LiYbF4", "LiYbF₄ (tetragonal)", "tetragonal", a=5.1335, c=10.588,
        space_group="I4₁/a (88)", reference="scheelite lit. (LiRF₄)", color="#fb8500"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Metric geometry (Cartesian embedding of the lattice)
# ═══════════════════════════════════════════════════════════════════════════
def crystal_to_cartesian(a: float, b: float, c: float,
                         alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return the 3×3 matrix whose *columns* are the direct lattice vectors a, b, c in
    a Cartesian frame (Å). Angles are in degrees. Uses the standard convention
    a ∥ x, b in the xy-plane."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz2 = c * c - cx * cx - cy * cy
    cz = np.sqrt(max(cz2, 0.0))
    vc = np.array([cx, cy, cz])
    return np.column_stack((va, vb, vc))


def reciprocal_matrix(direct: np.ndarray) -> np.ndarray:
    """Crystallographic reciprocal matrix R* (no 2π) whose *rows* are a*, b*, c*.

    Satisfies a*·a = 1, etc., so g(hkl) = R*ᵀ · [h, k, l] and |g| = 1/d (units 1/Å)."""
    return np.linalg.inv(direct)


@dataclass(frozen=True)
class Lattice:
    """Precomputed Cartesian geometry for one phase (all lengths Å / 1/Å)."""
    phase: Phase
    direct: np.ndarray        # columns a, b, c  (Å)
    recip: np.ndarray         # rows a*, b*, c*  (1/Å)

    @classmethod
    def from_phase(cls, phase: Phase) -> "Lattice":
        direct = crystal_to_cartesian(*phase.cell)
        return cls(phase=phase, direct=direct, recip=reciprocal_matrix(direct))

    def g_cart(self, hkl: Sequence[int]) -> np.ndarray:
        """Cartesian reciprocal vector g(hkl) in 1/Å."""
        return self.recip.T @ np.asarray(hkl, dtype=float)

    def d_spacing_A(self, hkl: Sequence[int]) -> float:
        """Interplanar spacing d(hkl) in Å."""
        g = self.g_cart(hkl)
        n = np.linalg.norm(g)
        return float("inf") if n == 0 else 1.0 / n

    def zone_axis_cart(self, uvw: Sequence[int]) -> np.ndarray:
        """Cartesian direction of the direct-lattice zone axis [uvw]."""
        return self.direct @ np.asarray(uvw, dtype=float)


def is_allowed(phase: Phase, hkl: Sequence[int]) -> bool:
    return _ABSENCE_RULES[phase.system](int(hkl[0]), int(hkl[1]), int(hkl[2]))


def d_spacing_nm(phase: Phase, hkl: Sequence[int]) -> float:
    """Convenience: d(hkl) in nanometres for a phase."""
    return Lattice.from_phase(phase).d_spacing_A(hkl) / 10.0


def to_miller_bravais(hkl: Sequence[int]) -> Tuple[int, int, int, int]:
    """3-index (hkl) → 4-index (hkil) for hexagonal reflections, i = -(h+k)."""
    h, k, l = int(hkl[0]), int(hkl[1]), int(hkl[2])
    return (h, k, -(h + k), l)


# ═══════════════════════════════════════════════════════════════════════════
# Reflection / zone-axis enumeration
# ═══════════════════════════════════════════════════════════════════════════
def _reduce_direction(vec: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
    """Reduce a lattice direction to its smallest integer form and canonical sign
    (so [uvw] and [-u-v-w] collapse to one). Returns None for the zero vector."""
    g = gcd(gcd(abs(vec[0]), abs(vec[1])), abs(vec[2]))
    if g == 0:
        return None
    v = (vec[0] // g, vec[1] // g, vec[2] // g)
    for comp in v:                       # canonical sign: first non-zero is positive
        if comp != 0:
            if comp < 0:
                v = (-v[0], -v[1], -v[2])
            break
    return v


def candidate_zone_axes(max_index: int = 2) -> List[Tuple[int, int, int]]:
    """Low-index zone axes [uvw] with |component| ≤ max_index, de-duplicated by
    direction. Covers the common viewing directions for all three systems."""
    seen = set()
    out: List[Tuple[int, int, int]] = []
    rng = range(-max_index, max_index + 1)
    for u, v, w in itertools.product(rng, rng, rng):
        red = _reduce_direction((u, v, w))
        if red is None or red in seen:
            continue
        seen.add(red)
        out.append(red)
    # Sort by sum of squares so the lowest-index axes ([001], [011], …) come first.
    out.sort(key=lambda t: (t[0] ** 2 + t[1] ** 2 + t[2] ** 2))
    return out


@dataclass
class Reflection:
    hkl: Tuple[int, int, int]
    g2d: np.ndarray          # 2-D reciprocal coord in the zone plane (1/nm)
    gmag: float              # |g| (1/nm)
    d_nm: float              # spacing (nm)


def zone_pattern(lattice: Lattice, uvw: Sequence[int],
                 d_min_nm: float, d_max_nm: float,
                 hkl_max: int = 6) -> List[Reflection]:
    """Predicted 2-D reciprocal net seen down zone axis [uvw].

    Returns the allowed reflections (Weiss zone law hu+kv+lw=0, plus the phase's
    systematic absences) with d in [d_min_nm, d_max_nm], each placed in an in-plane
    Cartesian frame so magnitudes and *relative* angles are exact. The absolute
    in-plane orientation is arbitrary (fixed later by matching to the image)."""
    u, v, w = int(uvw[0]), int(uvw[1]), int(uvw[2])
    t = lattice.zone_axis_cart((u, v, w))
    t_norm = np.linalg.norm(t)
    if t_norm == 0:
        return []
    t_hat = t / t_norm

    raw: List[Tuple[Tuple[int, int, int], np.ndarray, float]] = []
    rng = range(-hkl_max, hkl_max + 1)
    for h, k, l in itertools.product(rng, rng, rng):
        if h == 0 and k == 0 and l == 0:
            continue
        if h * u + k * v + l * w != 0:          # Weiss zone law: plane contains the axis
            continue
        if not _ABSENCE_RULES[lattice.phase.system](h, k, l):
            continue
        g = lattice.g_cart((h, k, l)) * 10.0    # 1/Å → 1/nm
        gmag = float(np.linalg.norm(g))
        if gmag == 0:
            continue
        d = 1.0 / gmag
        if not (d_min_nm <= d <= d_max_nm):
            continue
        raw.append(((h, k, l), g, gmag))

    if not raw:
        return []

    # In-plane orthonormal basis (e1, e2) ⊥ zone axis. e1 from the shortest g so it is
    # well-defined; e2 completes a right-handed frame about the zone axis.
    raw.sort(key=lambda r: r[2])
    e1 = raw[0][1] - np.dot(raw[0][1], t_hat) * t_hat
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(t_hat, e1)

    out: List[Reflection] = []
    for hkl, g, gmag in raw:
        g2d = np.array([np.dot(g, e1), np.dot(g, e2)])
        out.append(Reflection(hkl=hkl, g2d=g2d, gmag=gmag, d_nm=1.0 / gmag))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Matching a measured spot set to a predicted zone pattern
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SpotMatch:
    spot_index: int
    hkl: Tuple[int, int, int]
    d_meas_nm: float
    d_pred_nm: float
    rel_err: float


@dataclass
class Solution:
    phase: Phase
    uvw: Tuple[int, int, int]
    scale: float                       # predicted·scale ≈ measured (calibration factor)
    rotation_deg: float
    mirrored: bool
    matches: List[SpotMatch]
    n_matched: int
    n_spots: int
    mean_rel_err: float
    score: float

    @property
    def coverage(self) -> float:
        return self.n_matched / self.n_spots if self.n_spots else 0.0


def _rotmat(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def match_zone(measured: np.ndarray, predicted: List[Reflection],
               phase: Phase, uvw: Tuple[int, int, int],
               tol_frac: float = 0.06, min_matched: int = 3,
               n_ref_spots: int = 4) -> Optional[Solution]:
    """Align a predicted zone net to the measured spots by a similarity transform
    (scale + rotation, optional mirror) and score the agreement.

    ``measured`` is an (N,2) array of spot g-vectors in 1/nm (centred on DC). The
    scale that best maps predicted→measured is reported as a calibration factor.
    Returns the best Solution for this (phase, zone) or None if too few spots match."""
    n = len(measured)
    if n == 0 or not predicted:
        return None
    mags = np.linalg.norm(measured, axis=1)
    order = np.argsort(mags)[::-1]                 # strongest spots first as anchors
    ref_indices = [i for i in order if mags[i] > 0][:max(1, n_ref_spots)]

    pred_xy = np.array([r.g2d for r in predicted])
    pred_mag = np.array([r.gmag for r in predicted])

    best: Optional[Solution] = None
    for mi in ref_indices:
        m_vec = measured[mi]
        m_mag = mags[mi]
        if m_mag == 0:
            continue
        m_ang = np.arctan2(m_vec[1], m_vec[0])
        for pj in range(len(predicted)):
            if pred_mag[pj] == 0:
                continue
            scale = m_mag / pred_mag[pj]
            for mirror in (+1.0, -1.0):
                p_vec = pred_xy[pj] * np.array([1.0, mirror])
                p_ang = np.arctan2(p_vec[1], p_vec[0])
                rot = m_ang - p_ang
                R = _rotmat(rot)
                # Transform the whole predicted net into the measured frame.
                trans = (R @ (pred_xy * np.array([1.0, mirror]) * scale).T).T
                sol = _score_assignment(measured, mags, trans, predicted,
                                        phase, uvw, scale, np.degrees(rot),
                                        mirror < 0, tol_frac, min_matched)
                if sol is not None and (best is None or sol.score > best.score):
                    best = sol
    return best


def _score_assignment(measured: np.ndarray, mags: np.ndarray, trans: np.ndarray,
                      predicted: List[Reflection], phase: Phase,
                      uvw: Tuple[int, int, int], scale: float, rot_deg: float,
                      mirrored: bool, tol_frac: float,
                      min_matched: int) -> Optional[Solution]:
    """Greedy nearest-neighbour assignment of measured spots to transformed
    predicted reflections; build a Solution if enough spots match."""
    matches: List[SpotMatch] = []
    used_pred = set()
    for i in range(len(measured)):
        if mags[i] == 0:
            continue                              # skip DC
        tol = tol_frac * mags[i]
        dists = np.linalg.norm(trans - measured[i], axis=1)
        for pj in np.argsort(dists):
            if pj in used_pred:
                continue
            if dists[pj] > tol:
                break
            used_pred.add(int(pj))
            d_meas = 1.0 / mags[i]
            rel = abs(d_meas - predicted[pj].d_nm * scale) / (predicted[pj].d_nm * scale)
            matches.append(SpotMatch(
                spot_index=i, hkl=predicted[pj].hkl,
                d_meas_nm=d_meas, d_pred_nm=predicted[pj].d_nm, rel_err=rel))
            break

    n_matched = len(matches)
    if n_matched < min_matched:
        return None
    # Need at least two non-collinear matched spots for a real 2-D solution.
    if not _has_noncollinear_pair(measured, matches):
        return None

    mean_err = float(np.mean([m.rel_err for m in matches])) if matches else 1.0
    coverage = n_matched / max(1, int(np.count_nonzero(mags)))
    # Reward coverage and geometric fit; a bare-minimum 3-spot match should not
    # outrank a rich, low-error solution, so coverage dominates. Two phases of the
    # same crystal system that differ only by a ~1% isotropic lattice constant are
    # geometrically indistinguishable, so the fitted scale (agreement with the user's
    # nm/px calibration) breaks the tie: a soft penalty favours scale ≈ 1. Keep it
    # small so real geometry always outweighs calibration drift.
    scale_penalty = 0.15 * min(abs(scale - 1.0) / 0.20, 1.0)
    score = coverage * (1.0 - min(mean_err, 1.0)) + 0.01 * n_matched - scale_penalty
    return Solution(
        phase=phase, uvw=uvw, scale=scale, rotation_deg=rot_deg, mirrored=mirrored,
        matches=matches, n_matched=n_matched, n_spots=int(np.count_nonzero(mags)),
        mean_rel_err=mean_err, score=score)


def _has_noncollinear_pair(measured: np.ndarray, matches: List[SpotMatch]) -> bool:
    idx = [m.spot_index for m in matches]
    if len(idx) < 2:
        return False
    vecs = measured[idx]
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cross = vecs[i][0] * vecs[j][1] - vecs[i][1] * vecs[j][0]
            if abs(cross) > 1e-6 * (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-12):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Top-level: blind indexing across all phases and zone axes
# ═══════════════════════════════════════════════════════════════════════════
def index_pattern(measured: np.ndarray,
                  phases: Optional[Sequence[Phase]] = None,
                  tol_frac: float = 0.06,
                  max_zone_index: int = 2,
                  hkl_max: int = 6,
                  min_matched: int = 3,
                  top_n: int = 5) -> List[Solution]:
    """Blind index: search every phase × zone axis and return the best solutions.

    ``measured`` is an (N,2) array of spot g-vectors (1/nm) relative to the DC term.
    Solutions are ranked by score (coverage × geometric fit). The scale factor of the
    top solution is a calibration cross-check: ~1.0 means the nm/px scale is consistent
    with the identified phase."""
    if phases is None:
        phases = list(PHASES.values())
    measured = np.asarray(measured, dtype=float)
    mags = np.linalg.norm(measured, axis=1)
    nz = mags[mags > 0]
    if nz.size == 0:
        return []
    d_all = 1.0 / nz
    d_min = float(d_all.min()) * 0.6
    d_max = float(d_all.max()) * 1.6

    zones = candidate_zone_axes(max_zone_index)
    solutions: List[Solution] = []
    for phase in phases:
        lat = Lattice.from_phase(phase)
        for uvw in zones:
            pred = zone_pattern(lat, uvw, d_min, d_max, hkl_max=hkl_max)
            if len(pred) < 2:
                continue
            sol = match_zone(measured, pred, phase, uvw,
                             tol_frac=tol_frac, min_matched=min_matched)
            if sol is not None:
                solutions.append(sol)

    solutions.sort(key=lambda s: s.score, reverse=True)
    # Keep only the best zone axis per phase in the returned shortlist, but preserve
    # ranking so the overall winner stays on top.
    seen_phase = set()
    shortlist: List[Solution] = []
    for s in solutions:
        tag = (s.phase.key, s.uvw)
        if tag in seen_phase:
            continue
        seen_phase.add(tag)
        shortlist.append(s)
        if len(shortlist) >= top_n:
            break
    return shortlist


def interplanar_angle_deg(phase: Phase, hkl1: Sequence[int], hkl2: Sequence[int]) -> float:
    """Angle between two plane normals (reciprocal vectors) in degrees."""
    lat = Lattice.from_phase(phase)
    g1, g2 = lat.g_cart(hkl1), lat.g_cart(hkl2)
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = np.clip(np.dot(g1, g2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))
