# Day 4 — Cross‑Sectional Momentum (12‑1)
![Equity Curve](charts/xs_mom_equity.png)

Go long the **top‑N ETFs** ranked by **12‑1 momentum** each month. Equal‑weight picks, pay **transaction costs**, subtract **cash drag**, compare vs **equal‑weight** and **SPY**.

## Features
- Universe: sectors + TLT/GLD by default (customizable)
- Signal: 12‑1 momentum with 1‑month skip to avoid look‑ahead
- Monthly/quarterly rebalancing with `safe_freq` to avoid pandas warnings
- Transaction costs & cash drag
- Outputs: CSVs + equity curve PNG

## Quickstart
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python xs_momentum.py

## Outputs
- outputs/daily_returns.csv
- outputs/equity_curves.csv
- outputs/weights_daily.csv
- outputs/picks_by_rebalance.csv
- charts/xs_mom_equity.png
```




