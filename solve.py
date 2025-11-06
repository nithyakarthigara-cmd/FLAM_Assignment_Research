"""
solve.py
----------------------------------------
This script estimates the parameters θ (theta), M, and X
for a nonlinear parametric curve that fits given (x, y) data.

It minimizes the L1 distance (sum of absolute errors)
between the predicted and actual data using SciPy optimization.
----------------------------------------
Author: Nithyakarthiga R
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---------- Configuration ----------
CSV_PATH = r"csv file pathh"   # Change if needed
OUT_DIR = r"output file path"
os.makedirs(OUT_DIR, exist_ok=True)

# Bounds for parameters: (theta, M, X)
BOUNDS = [(0.0, 50.0), (-0.05, 0.05), (0.0, 100.0)]
INITIAL_GUESS = [25.0, 0.01, 50.0]
# -----------------------------------


# ---------- Load CSV ----------
df = pd.read_csv(CSV_PATH)
if list(df.columns)[:2] != ['x', 'y']:
    df = df.rename(columns={df.columns[0]:'x', df.columns[1]:'y'})

x_actual = df['x'].values
y_actual = df['y'].values
n = len(df)
t = np.linspace(6, 60, n)   # t uniformly spaced in (6, 60)
# ------------------------------


# ---------- Define Curve ----------
def curve(params, tvals):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)
    exp_term = np.exp(M * np.abs(tvals))
    x_pred = tvals * np.cos(theta) - exp_term * np.sin(0.3 * tvals) * np.sin(theta) + X
    y_pred = 42 + tvals * np.sin(theta) + exp_term * np.sin(0.3 * tvals) * np.cos(theta)
    return x_pred, y_pred
# ----------------------------------


# ---------- Define L1 Cost ----------
def l1_cost(params):
    x_pred, y_pred = curve(params, t)
    return np.sum(np.abs(x_pred - x_actual) + np.abs(y_pred - y_actual))
# ------------------------------------


# ---------- Optimization ----------
res = minimize(l1_cost, INITIAL_GUESS, bounds=BOUNDS, method="L-BFGS-B", options={'maxiter': 10000})
theta_opt, M_opt, X_opt = res.x
cost_val = float(res.fun)
# ----------------------------------


# ---------- Save Results ----------
params_out = {
    "theta_deg": float(theta_opt),
    "M": float(M_opt),
    "X": float(X_opt),
    "L1_cost": cost_val,
    "n_points": int(n)
}
with open(os.path.join(OUT_DIR, "fitted_params.json"), "w") as f:
    json.dump(params_out, f, indent=4)

x_pred, y_pred = curve([theta_opt, M_opt, X_opt], t)
pred_df = pd.DataFrame({
    "t": t,
    "x_pred": x_pred,
    "y_pred": y_pred,
    "x_actual": x_actual,
    "y_actual": y_actual
})
pred_df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
# ----------------------------------


# ---------- Plots ----------
plt.figure(figsize=(8,5))
plt.scatter(x_actual, y_actual, s=10, label='Actual', alpha=0.7)
plt.scatter(x_pred, y_pred, s=8, label='Predicted', alpha=0.6)
plt.legend()
plt.title("Actual (blue) vs Predicted (orange) points")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "xy_compare.png"), dpi=200)
plt.close()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.scatter(t, x_actual - x_pred, s=8)
plt.axhline(0, linestyle='--')
plt.title("Residuals: x_actual - x_pred")
plt.xlabel("t")

plt.subplot(1,2,2)
plt.scatter(t, y_actual - y_pred, s=8)
plt.axhline(0, linestyle='--')
plt.title("Residuals: y_actual - y_pred")
plt.xlabel("t")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residuals.png"), dpi=200)
plt.close()
# ----------------------------------


# ---------- Text Report ----------
with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
    f.write("Fitted parameters (assignment bounds respected):\n")
    f.write(f"Theta (deg): {theta_opt:.6f}\n")
    f.write(f"M: {M_opt:.6f}\n")
    f.write(f"X: {X_opt:.6f}\n")
    f.write(f"L1 cost: {cost_val:.6f}\n\n")
    f.write("Notes:\n")
    f.write("- t generated uniformly in (6, 60)\n")
    f.write("- L1 distance minimized using scipy.optimize.minimize (L-BFGS-B)\n")
    f.write("- Outputs saved in 'outputs/' directory\n")
# ----------------------------------


print("DONE. Outputs saved to:", OUT_DIR)
print(json.dumps(params_out, indent=2))
