# FLAM Assignment — Research and Development (AI)

This repository contains my FLAM placement assignment:  
parameter estimation of a nonlinear parametric curve using **L1 optimization**.

##  Objective
Estimate parameters **θ**, **M**, and **X** so that the predicted curve
best fits the experimental data (`x`, `y`) provided in `xy_data.csv`.

##  Methodology
1. Load `xy_data.csv`
2. Generate `t` values in range (6, 60)
3. Define equations:
x(t) = t*cos(θ) - e^(M|t|)*sin(0.3t)sin(θ) + X
y(t) = 42 + tsin(θ) + e^(M|t|)*sin(0.3t)*cos(θ)
4. Minimize the **L1 distance** between predicted and actual values using `scipy.optimize.minimize`
5. Save results and visualizations in `outputs/`

## Results 
| Parameter | Value |
|------------|--------|
| θ (deg)    | 28.12 |
| M          | 0.0214 |
| X          | 54.90 |
| L1 cost    | 37865.09 |

## Requirements
pip install numpy pandas matplotlib scipy
