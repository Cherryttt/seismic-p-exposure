from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass(frozen=True)
class OmoriFit:
    K: float
    c: float
    p: float
    nll: float
    tmin: float
    tmax: float
    n_events: int

    def to_dict(self) -> Dict:
        return asdict(self)


def _safe_log(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log(np.maximum(x, 1e-300))


def omori_nll(params: NDArray[np.float64], t: NDArray[np.float64], tmin: float, tmax: float) -> float:
    # params = [logK, logc, logp] to enforce positivity
    logK, logc, logp = params
    K = float(np.exp(logK))
    c = float(np.exp(logc))
    p = float(np.exp(logp))

    # log-likelihood for non-homogeneous Poisson process:
    # L = sum log lambda(t_i) - integral_{tmin}^{tmax} lambda(t) dt
    # lambda(t) = K / (t + c)^p
    # integral depends on p:
    # if p != 1:  K * ((tmax+c)^(1-p) - (tmin+c)^(1-p)) / (1-p)
    # if p == 1:  K * log((tmax+c)/(tmin+c))

    tt = t
    lam_log = math.log(K) - p * _safe_log(tt + c)
    ll1 = float(np.sum(lam_log))

    if abs(p - 1.0) > 1e-6:
        integral = K * ((tmax + c) ** (1.0 - p) - (tmin + c) ** (1.0 - p)) / (1.0 - p)
    else:
        integral = K * math.log((tmax + c) / (tmin + c))

    ll = ll1 - float(integral)
    return -ll


def fit_omori_mle(
    t: NDArray[np.float64],
    *,
    tmin: float,
    tmax: float,
    init: Optional[Tuple[float, float, float]] = None,
) -> OmoriFit:
    t = np.asarray(t, dtype=np.float64)
    t = t[(t >= tmin) & (t <= tmax)]
    if t.size < 20:
        raise ValueError(f"Too few events for fit: n={t.size}")

    if init is None:
        # Rough init: p ~ 1.1, c small, K proportional to count.
        p0 = 1.1
        c0 = max(1e-4, tmin / 10.0)
        # Solve K from expected count approx at p0/c0:
        if abs(p0 - 1.0) > 1e-6:
            denom = ((tmax + c0) ** (1.0 - p0) - (tmin + c0) ** (1.0 - p0)) / (1.0 - p0)
        else:
            denom = math.log((tmax + c0) / (tmin + c0))
        K0 = max(1e-6, t.size / max(denom, 1e-6))
    else:
        K0, c0, p0 = init

    x0 = np.log([K0, c0, p0]).astype(np.float64)

    res = minimize(
        omori_nll,
        x0=x0,
        args=(t, float(tmin), float(tmax)),
        method="L-BFGS-B",
        # Bound log-params to keep the optimization numerically stable and
        # prevent pathological fits (e.g. p -> huge) when the catalog is sparse.
        bounds=[
            (-30.0, 30.0),  # logK
            (math.log(1e-6), math.log(10.0)),  # logc, in days
            (math.log(0.2), math.log(5.0)),  # logp
        ],
        options={"maxiter": 2000},
    )
    if not res.success:
        raise RuntimeError(f"MLE failed: {res.message}")

    logK, logc, logp = res.x
    K, c, p = float(np.exp(logK)), float(np.exp(logc)), float(np.exp(logp))
    return OmoriFit(K=K, c=c, p=p, nll=float(res.fun), tmin=float(tmin), tmax=float(tmax), n_events=int(t.size))


def bootstrap_p(
    t: NDArray[np.float64],
    *,
    tmin: float,
    tmax: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    t = np.asarray(t, dtype=np.float64)
    t = t[(t >= tmin) & (t <= tmax)]
    if t.size < 20:
        return np.array([], dtype=np.float64)

    ps: List[float] = []
    for _ in range(n):
        samp = rng.choice(t, size=t.size, replace=True)
        try:
            fit = fit_omori_mle(samp, tmin=tmin, tmax=tmax)
            ps.append(fit.p)
        except Exception:
            continue
    return np.asarray(ps, dtype=np.float64)


def binned_rate(t: NDArray[np.float64], tmin: float, tmax: float, n_bins: int = 30) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Log-spaced bins and rate per day.
    edges = np.logspace(np.log10(tmin), np.log10(tmax), n_bins + 1)
    counts, _ = np.histogram(t, bins=edges)
    widths = edges[1:] - edges[:-1]
    centers = np.sqrt(edges[1:] * edges[:-1])
    rate = counts / widths
    return centers, rate
