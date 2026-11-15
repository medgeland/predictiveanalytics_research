# Global LightGBM vs Croston's method for daily product-store sales (M5)

Reproducible code for the Final Research Project, Predictive Analytics Methods.
Author: Margarita Serebrennikova (AIB-2401)

## What it does
Compares three forecasting models (seasonal naive, Croston, one global LightGBM
per store) for the daily unit sales of 3,049 products in three Walmart stores
(TX_1, CA_1, WI_1) from the M5 dataset, using three rolling 28-day test windows
(29 Feb - 22 May 2016). It reports RMSSE per product, tests RQ1 (is LightGBM
more accurate?) and RQ2 (does the gain differ by demand pattern?), and scores
forecasts summed to department and store totals.

## Data
Download these three files from the M5 Forecasting - Accuracy competition on
Kaggle (https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) into
one folder, e.g. `m5/`:
- calendar.csv
- sales_train_evaluation.csv
- sell_prices.csv

## Run
    pip install -r requirements.txt
    python m5_final_analysis.py --data-dir m5 --out-dir results

Runtime: about 11 minutes on a 2-core machine with 8 GB RAM. A fixed seed (42)
and LightGBM's deterministic mode make repeated runs identical.

## Outputs (results/)
| File | Contents |
|---|---|
| summary.json | Every statistic reported in the paper |
| product_results.csv | RMSSE of each model for each product, store and test window |
| product_store_mean_r.csv | Log ratio r averaged over the 3 windows (RQ1/RQ2 tests) |
| department_results.csv | RMSSE of department totals |
| store_results.csv | RMSSE of store totals |
| eda_monthly_sales.csv | Monthly average daily sales (Figure 1) |

## Leakage controls
- Every model uses only data before its forecast origin.
- All LightGBM sales features are lagged >= 28 days (the horizon).
- Demand patterns use data before the first test window (29 Feb 2016).
- Number of trees and best benchmark are chosen on a validation window; each
  test window is scored once.
- All dates are read from calendar.csv; the script asserts that the last test
  window ends on the last day of the file.
