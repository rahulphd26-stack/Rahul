# Prediction of Ballistic Impact Behaviour of Kevlar/Basalt Hybrid Epoxy Composite for Body Armor using AI-ML

ANSYS 2026 R1 Explicit Dynamics (AUTODYN) simulations of a hemispherical projectile hitting a
150 x 150 x 3 mm Kevlar/Basalt/Epoxy laminate, plus machine-learning surrogates
(Linear Regression, Random Forest, Support Vector Regression) that predict the stress- and
strain-based response from layer stacking and impact velocity.

## Repository layout
```
data/ballistic_dataset.csv     16 FEA runs (4 stackings x 4 velocities: 350-500 m/s)
src/run_experiments.py         full ML pipeline (training, validation, figures, tables)
results/                       generated tables (model_comparison.csv, rf_feature_importance.csv, ...)
figures/                       generated plots; figures/fea/ = ANSYS screenshots
paper/paper.tex                IEEE-format paper (upload folder to Overleaf)
paper/paper_preview.pdf        local preview (approximate layout; Overleaf gives true IEEEtran layout)
```

## Run
```bash
pip install -r requirements.txt
python src/run_experiments.py      # ~2-3 min on a laptop CPU; fixed seed (42)
```
Outputs: `results/model_comparison.csv`, `results/rf_feature_importance.csv`,
`results/layup_deformation_table.csv`, and the PNG figures in `figures/`.

## Dataset columns
`layer1, layer2, layer3` (layer material from impacted face), `velocity_ms`,
`total_deformation_mm`, `von_mises_stress_mpa`, `directional_deformation_mm`,
`equivalent_elastic_strain` (mm/mm).

## Method summary
* Inputs: one-hot encoded layer materials + standardised velocity. One model per response.
* Models: Linear Regression, Random Forest (100 trees), SVR (linear/RBF, C and epsilon tuned
  by inner 4-fold grid search; inputs and targets standardised).
* Validation (only 16 samples): leave-one-out CV, 4-fold CV x 5 repeats, and
  leave-one-velocity-out (unseen-velocity test).

## Key results (LOOCV R2)
| Response | Linear Reg. | Random Forest | SVR |
|---|---|---|---|
| Total deformation | 0.9969 | 0.9247 | 0.9968 |
| Von Mises stress | 0.9996 | 0.9330 | 0.9996 |
| Directional deformation | 0.9953 | 0.9079 | 0.9952 |
| Equivalent elastic strain | 0.9982 | 0.9157 | 0.9978 |

Random Forest drops to R2 ~ 0.20 on the unseen-velocity test (trees cannot extrapolate);
Linear Regression and SVR stay above 0.99. Velocity accounts for 92-96 % of RF importance.
Kevlar/Epoxy/Basalt gives the lowest deformation, stress and strain at every velocity.

## Limitations
Small dataset (16 runs, one plate thickness, one projectile); no mesh-convergence study,
experimental validation or residual-velocity/ballistic-limit analysis; weight and cost are not
quantified. The surrogates are valid only inside the sampled design space.

## Authors
[Add names / roll numbers]
