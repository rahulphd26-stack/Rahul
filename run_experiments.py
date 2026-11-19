"""Predict ballistic response of Kevlar/Basalt hybrid epoxy laminates from ANSYS data.

Models: Linear Regression, Random Forest, Support Vector Regression (SVR).
Validation (16 samples only -> no single train/test split):
  * Leave-one-out CV (LOOCV): pooled out-of-sample R2 / MAE / RMSE
  * Repeated 4-fold CV (10 repeats): mean +- std of R2
  * Leave-one-velocity-out: hold out one impact velocity (all layups), i.e. a
    harder "unseen velocity" test
SVR uses standardised inputs and targets and a hyper-parameter grid searched in
an inner CV loop (nested CV), so no test information leaks into tuning.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, LeaveOneOut, LeaveOneGroupOut,
                                     RepeatedKFold, cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FEATURES = ["layer1", "layer2", "layer3", "velocity_ms"]
TARGETS = {
    "total_deformation_mm": "Total deformation (mm)",
    "von_mises_stress_mpa": "Von-Mises stress (MPa)",
    "directional_deformation_mm": "Directional deformation (mm)",
    "equivalent_elastic_strain": "Equivalent elastic strain (mm/mm)",
}


def preprocessor():
    # Full layer-stack encoding (all categories kept; no dropped dummy needed for RF/SVR,
    # tiny ridge-free LR copes with collinearity via lstsq min-norm solution).
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["layer1", "layer2", "layer3"]),
        ("num", StandardScaler(), ["velocity_ms"]),
    ])


def make_models():
    lr = Pipeline([("prep", preprocessor()), ("m", LinearRegression())])
    rf = Pipeline([("prep", preprocessor()),
                   ("m", RandomForestRegressor(n_estimators=100, random_state=SEED))])
    svr_inner = Pipeline([("prep", preprocessor()), ("m", SVR(kernel="rbf"))])
    grid = {"m__kernel": ["linear", "rbf"], "m__C": [1, 10, 100, 1000],
            "m__epsilon": [0.01, 0.1]}
    svr = GridSearchCV(svr_inner, grid, cv=KFold(4, shuffle=True, random_state=SEED),
                       scoring="r2", n_jobs=1)
    scaled_svr = TransformedTargetRegressor(regressor=svr, transformer=StandardScaler())
    # Un-tuned, un-scaled-target SVR reproduces the weak baseline of the first draft
    naive = Pipeline([("m", SVR())])
    return {"Linear Regression": lr, "Random Forest": rf, "SVR (tuned, scaled)": scaled_svr}, naive


def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))))


def main():
    df = pd.read_csv(ROOT / "data" / "ballistic_dataset.csv")
    X = df[FEATURES]
    models, _ = make_models()
    rows, preds = [], {}
    for tgt in TARGETS:
        y = df[tgt]
        for name, mdl in models.items():
            p = cross_val_predict(mdl, X, y, cv=LeaveOneOut())
            preds[(tgt, name)] = p
            m = metrics(y, p)
            rkf = cross_val_score(mdl, X, y, scoring="r2",
                                  cv=RepeatedKFold(n_splits=4, n_repeats=5, random_state=SEED))
            pv = cross_val_predict(mdl, X, y, cv=LeaveOneGroupOut(), groups=df["velocity_ms"])
            mv = metrics(y, pv)
            rows.append(dict(target=tgt, model=name, **{f"LOOCV_{k}": v for k, v in m.items()},
                             RKF_R2_mean=rkf.mean(), RKF_R2_std=rkf.std(),
                             LOVO_R2=mv["R2"], LOVO_MAE=mv["MAE"], LOVO_RMSE=mv["RMSE"]))
            print(f"{tgt:28s} {name:20s} LOOCV R2={m['R2']:.4f}  RKF R2={rkf.mean():.4f}  LOVO R2={mv['R2']:.4f}")
    res = pd.DataFrame(rows)
    out = ROOT / "results"
    res.round(4).to_csv(out / "model_comparison.csv", index=False)

    # Feature importance (RF fitted on all data) + linear velocity sensitivity
    imp = {}
    for tgt in TARGETS:
        rf = models["Random Forest"].fit(X, df[tgt])
        names = rf.named_steps["prep"].get_feature_names_out()
        imp[tgt] = dict(zip(names, rf.named_steps["m"].feature_importances_.round(4)))
    pd.DataFrame(imp).to_csv(out / "rf_feature_importance.csv")

    # Layup ranking at each velocity (measured, from FEA)
    df["layup"] = df.layer1 + "/" + df.layer2 + "/" + df.layer3
    df.pivot(index="layup", columns="velocity_ms", values="total_deformation_mm").to_csv(out / "layup_deformation_table.csv")

    # ---- figures ----
    fig_dir = ROOT / "figures"
    plt.rcParams.update({"font.size": 9})
    piv = res.pivot(index="target", columns="model", values="LOOCV_R2").loc[list(TARGETS)]
    ax = piv.plot(kind="bar", figsize=(6.5, 3.4), width=0.8)
    ax.set_xticklabels([TARGETS[t].split(" (")[0] for t in piv.index], rotation=15, ha="right")
    ax.set_ylabel("LOOCV $R^2$"); ax.set_xlabel(""); ax.axhline(0, color="k", lw=.6)
    ax.legend(fontsize=7, loc="lower left"); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(fig_dir / "model_comparison_r2.png", dpi=300); plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.2))
    for ax, tgt in zip(axs.ravel(), TARGETS):
        for name, mk in zip(models, ["o", "s", "^"]):
            ax.scatter(df[tgt], preds[(tgt, name)], s=18, marker=mk, label=name, alpha=.8)
        lo, hi = df[tgt].min(), df[tgt].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8)
        ax.set_title(TARGETS[tgt], fontsize=8); ax.set_xlabel("FEA"); ax.set_ylabel("Predicted (LOOCV)")
    axs[0, 0].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(fig_dir / "parity_plots.png", dpi=300); plt.close()

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for lay, g in df.groupby("layup"):
        ax.plot(g.velocity_ms, g.total_deformation_mm, "o-", label=lay)
    ax.set_xlabel("Impact velocity (m/s)"); ax.set_ylabel("Total deformation (mm)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(fig_dir / "deformation_vs_velocity.png", dpi=300); plt.close()

    sub = pd.DataFrame(imp)["total_deformation_mm"].sort_values()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.barh([s.replace("cat__", "").replace("num__", "") for s in sub.index], sub.values)
    ax.set_xlabel("RF importance (total deformation)")
    plt.tight_layout(); plt.savefig(fig_dir / "rf_importance.png", dpi=300); plt.close()
    print("Done. Outputs in results/ and figures/")


if __name__ == "__main__":
    main()
