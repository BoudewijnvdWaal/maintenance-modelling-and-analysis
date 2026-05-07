import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_train(data_dir: Path) -> pd.Series:
    fp = data_dir / "train_FD001.txt"
    cols = ["engine_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    df = pd.read_csv(fp, sep=r"\s+", header=None, names=cols)
    lifetimes = df.groupby("engine_id")["cycle"].max()
    return lifetimes


def fit_distribution(data: np.ndarray, dist_name: str):
    # Map name to scipy distribution
    dists = {
        "weibull": stats.weibull_min,
        "lognormal": stats.lognorm,
        "exponential": stats.expon,
        "gamma": stats.gamma,
    }
    dist = dists[dist_name]
    params = dist.fit(data)

    # log-likelihood
    ll = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * ll
    return {"dist": dist, "params": params, "ll": ll, "aic": aic}


def survival(dist, params, t: np.ndarray) -> np.ndarray:
    return 1.0 - dist.cdf(t, *params)


def hazard(dist, params, t: np.ndarray) -> np.ndarray:
    pdf = dist.pdf(t, *params)
    sf = 1.0 - dist.cdf(t, *params)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(sf > 0, pdf / sf, np.nan)
    return h


def expected_min_T(dist, params, t: float) -> float:
    # E[min(T, t)] = 
    # = 
    f = lambda u: 1.0 - dist.cdf(u, *params)
    val, _ = quad(f, 0, t, limit=200)
    return val


def g_of_t(dist, params, t: float, C_p: float, C_c: float) -> float:
    S_t = 1.0 - dist.cdf(t, *params)
    num = C_p * S_t + C_c * (1 - S_t)
    den = expected_min_T(dist, params, t)
    if den <= 0:
        return np.inf
    return num / den


def main():
    try:
        lifetimes = load_train(DATA_DIR)
    except Exception as e:
        print("Error loading training data:", e, file=sys.stderr)
        sys.exit(1)

    data = lifetimes.values.astype(float)
    print(f"Loaded {len(data)} engine lifetimes. range: {data.min()} - {data.max()}")

    candidates = ["weibull", "lognormal", "exponential", "gamma"]
    fits = {}
    for name in candidates:
        try:
            fits[name] = fit_distribution(data, name)
        except Exception as e:
            print(f"Failed fitting {name}: {e}")

    print("Fit results (AIC):")
    for name, r in fits.items():
        print(f"- {name}: AIC={r['aic']:.1f}, ll={r['ll']:.1f}, params={r['params']}")

    best = min(fits.items(), key=lambda kv: kv[1]["aic"])
    best_name, best_res = best[0], best[1]
    print(f"\nSelected best distribution: {best_name}")

    # Plot hazard
    dist = best_res["dist"]
    params = best_res["params"]
    tgrid = np.linspace(1, data.max() * 1.1, 200)
    h = hazard(dist, params, tgrid)
    plt.figure()
    plt.plot(tgrid, h)
    plt.xlabel("t (cycles)")
    plt.ylabel("hazard h(t)")
    plt.title(f"Hazard function - {best_name}")
    plt.grid(True)
    plt.savefig(OUT_DIR / "hazard.png", dpi=150)
    print(f"Saved hazard plot to {OUT_DIR / 'hazard.png'}")

    # Compute optimal t
    C_p = 10000.0
    C_c = 100000.0

    tmax = float(max(data.max() * 1.5, 1.0))

    # coarse grid search
    grid = np.linspace(1, tmax, 200)
    gvals = [g_of_t(dist, params, tt, C_p, C_c) for tt in grid]
    idx = int(np.nanargmin(gvals))
    t0 = grid[idx]

    # refine
    res = minimize_scalar(lambda tt: g_of_t(dist, params, tt, C_p, C_c), bounds=(1, tmax), method="bounded")
    t_opt = res.x if res.success else t0
    g_opt = g_of_t(dist, params, t_opt, C_p, C_c)

    print(f"Optimal preventive replacement time t*: {t_opt:.2f} cycles")
    print(f"Long-run average cost g(t*): {g_opt:.2f} (currency units per cycle)")

    # plot g(t)
    grid_fine = np.linspace(1, tmax, 300)
    gfine = [g_of_t(dist, params, tt, C_p, C_c) for tt in grid_fine]
    plt.figure()
    plt.plot(grid_fine, gfine)
    plt.axvline(t_opt, color="red", linestyle="--", label=f"t*={t_opt:.1f}")
    plt.xlabel("t (cycles)")
    plt.ylabel("g(t) - avg cost per cycle")
    plt.title("Average cost g(t) vs t")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / "g_of_t.png", dpi=150)
    print(f"Saved g(t) plot to {OUT_DIR / 'g_of_t.png'}")


if __name__ == "__main__":
    main()
