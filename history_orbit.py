"""
speckle_suite.history_orbit
============================
Keplerian orbit computation for the History tab.

Public API
----------
solve_kepler()        vectorised Kepler's equation solver (E from M)
compute_orbit_curve() full ellipse in (theta, rho) from ORB6 elements
                      or interpolated ephemeris as fallback
"""

from __future__ import annotations

import numpy as np


# ── Kepler solver ──────────────────────────────────────────────────────────

def solve_kepler(
        M: np.ndarray,
        e: float,
        tol: float = 1e-10,
) -> np.ndarray:
    """
    Solve Kepler's equation  M = E − e·sin(E)  for E, vectorised.

    Uses Newton-Raphson iteration; converges in < 10 steps for e < 0.99.

    Parameters
    ----------
    M   : mean anomaly array [radians]
    e   : eccentricity  0 ≤ e < 1
    tol : convergence tolerance

    Returns
    -------
    E : eccentric anomaly array [radians]
    """
    E = M.copy()
    for _ in range(50):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


# ── Orbital ellipse ────────────────────────────────────────────────────────

def compute_orbit_curve(
        ephem_pts: list[tuple[float, float, float]],
        orb_elem: dict | None = None,
        n_pts: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the projected orbital ellipse in (theta_deg, rho_arcsec).

    Primary path: Keplerian elements via Thiele-Innes constants.
    Fallback:     interpolated ephemeris when elements are unavailable.

    Parameters
    ----------
    ephem_pts : [(year, theta_deg, rho_arcsec), …] from query_orb6_ephem()
    orb_elem  : dict from query_orb6_elements(), or None
    n_pts     : number of sample points around the orbit

    Returns
    -------
    theta_arr : (n_pts,) float64  position angle [degrees]
    rho_arr   : (n_pts,) float64  separation [arcseconds]
    """
    if orb_elem and orb_elem.get("P") and orb_elem.get("a"):
        a     = orb_elem["a"]
        e     = orb_elem.get("e",     0.0)
        i     = np.radians(orb_elem.get("i",     0.0))
        Omega = np.radians(orb_elem.get("Omega", 0.0))
        omega = np.radians(orb_elem.get("omega", 0.0))

        # Thiele-Innes constants (Heintz convention)
        A =  a * ( np.cos(omega) * np.cos(Omega) - np.sin(omega) * np.sin(Omega) * np.cos(i))
        B =  a * ( np.cos(omega) * np.sin(Omega) + np.sin(omega) * np.cos(Omega) * np.cos(i))
        F =  a * (-np.sin(omega) * np.cos(Omega) - np.cos(omega) * np.sin(Omega) * np.cos(i))
        G =  a * (-np.sin(omega) * np.sin(Omega) + np.cos(omega) * np.cos(Omega) * np.cos(i))

        # Sample uniformly in mean anomaly
        M_arr = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        E_arr = solve_kepler(M_arr, e)
        X     = np.cos(E_arr) - e
        Y     = np.sqrt(1 - e**2) * np.sin(E_arr)
        dRA   = B * X + G * Y
        dDec  = A * X + F * Y
        return np.degrees(np.arctan2(dRA, dDec)) % 360, np.hypot(dRA, dDec)

    # ── Ephemeris fallback ────────────────────────────────────────────────
    if len(ephem_pts) < 2:
        return (np.array([p[1] for p in ephem_pts]),
                np.array([p[2] for p in ephem_pts]))

    from scipy.interpolate import interp1d
    years  = np.array([p[0] for p in ephem_pts])
    thetas = np.unwrap(np.radians([p[1] for p in ephem_pts]))
    rhos   = np.array([p[2] for p in ephem_pts])
    t_int  = np.linspace(years[0], years[-1], n_pts)
    return (
        np.degrees(interp1d(years, thetas, fill_value="extrapolate")(t_int)) % 360,
        interp1d(years, rhos, fill_value="extrapolate")(t_int),
    )
