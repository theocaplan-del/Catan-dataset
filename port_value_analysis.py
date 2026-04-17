"""
Port value analysis: what is a 3:1 port worth in pip terms at initial placement?

Method:
  - Use turn-10 snapshots (closest to post-initial-placement board state)
  - Logistic regression: logit(win) ~ total_pip + port_generic + turn_order
  - Port value in pip units = coef(port_generic) / coef(total_pip)
  - Validated by direct win-rate binning across pip buckets
  - Also checks 2:1 specific ports for comparison
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("features_full.csv")

# Use the first snapshot per game (closest to post-initial-placement state)
# First snapshot is emitted at completedTurns=9 due to the -1 seed in extraction
first_turn = df.groupby("game_id")["turn"].transform("min")
snap = df[df["turn"] == first_turn].copy()
print(f"First-snapshot rows: {len(snap):,}  ({snap['game_id'].nunique():,} games)"
      f"  median turn = {snap['turn'].median():.0f}\n")

# How many players have each port type at turn 10?
print("Port access rates at turn 10:")
for col, label in [
    ("port_generic", "3:1 generic"),
    ("port_brick",   "2:1 brick"),
    ("port_grain",   "2:1 grain"),
    ("port_wool",    "2:1 wool"),
    ("port_ore",     "2:1 ore"),
    ("port_lumber",  "2:1 lumber"),
]:
    n = snap[col].sum()
    pct = n / len(snap) * 100
    wr  = snap.loc[snap[col] == 1, "is_winner"].mean() * 100
    print(f"  {label:>15}: {n:>7,} ({pct:>4.1f}% of players)  win rate {wr:.1f}%")
print(f"  {'none':>15}: {(snap[['port_generic','port_brick','port_grain','port_wool','port_ore','port_lumber']].sum(axis=1)==0).sum():>7,}  win rate {snap.loc[(snap[['port_generic','port_brick','port_grain','port_wool','port_ore','port_lumber']].sum(axis=1)==0), 'is_winner'].mean()*100:.1f}%")
print(f"\n  Baseline win rate: 25.0%\n")

# ── Pip distribution by port status ───────────────────────────────────────────
print("=" * 62)
print("Pip score distribution: 3:1 port vs no port")
print("-" * 62)
has_port = snap["port_generic"] == 1
no_port  = snap["port_generic"] == 0
print(f"  With 3:1 port: mean pip = {snap.loc[has_port, 'total_pip'].mean():.1f}  "
      f"median = {snap.loc[has_port, 'total_pip'].median():.0f}")
print(f"  Without port:  mean pip = {snap.loc[no_port, 'total_pip'].mean():.1f}  "
      f"median = {snap.loc[no_port, 'total_pip'].median():.0f}")
print(f"  Average pip penalty from taking a 3:1 port corner: "
      f"{snap.loc[no_port, 'total_pip'].mean() - snap.loc[has_port, 'total_pip'].mean():.1f}\n")

# ── Logistic regression to isolate port effect ────────────────────────────────
print("=" * 62)
print("Logistic regression: win ~ total_pip + port_generic + turn_order")
print("(unscaled coefficients — pip effect in log-odds per 1-pip change)")
print("-" * 62)

X = snap[["total_pip", "port_generic", "turn_order"]].values
y = snap["is_winner"].values

lr = LogisticRegression(max_iter=1000, C=1e6)  # high C = minimal regularisation
lr.fit(X, y)

coef_pip  = lr.coef_[0][0]
coef_port = lr.coef_[0][1]
coef_turn = lr.coef_[0][2]

print(f"  total_pip coef:    {coef_pip:+.5f}  (log-odds per pip)")
print(f"  port_generic coef: {coef_port:+.5f}  (log-odds for having 3:1 port)")
print(f"  turn_order coef:   {coef_turn:+.5f}")
print()

port_pip_equiv = coef_port / coef_pip
print(f"  3:1 port ≈  {port_pip_equiv:.1f} pip-equivalent  (coef_port / coef_pip)")

# Convert to probability at 25% baseline
p0   = 0.25
dp_pip  = coef_pip  * p0 * (1 - p0)   # dP per pip at baseline
dp_port = coef_port * p0 * (1 - p0)   # dP for port at baseline
print(f"  At the 25% baseline:")
print(f"    Each additional pip  → +{dp_pip*100:+.3f}pp win rate")
print(f"    Having 3:1 port      → +{dp_port*100:+.3f}pp win rate")
print()

# ── Also fit all specific port types together ─────────────────────────────────
print("=" * 62)
print("All port types: pip-equivalent comparison")
print("-" * 62)
port_cols = ["port_generic", "port_brick", "port_grain", "port_wool", "port_ore", "port_lumber"]
port_labels = {"port_generic": "3:1 generic", "port_brick": "2:1 brick",
               "port_grain": "2:1 grain", "port_wool": "2:1 wool",
               "port_ore": "2:1 ore", "port_lumber": "2:1 lumber"}
X2 = snap[["total_pip"] + port_cols + ["turn_order"]].values
lr2 = LogisticRegression(max_iter=1000, C=1e6)
lr2.fit(X2, y)
pip_coef2 = lr2.coef_[0][0]
print(f"  {'Port':>15}  {'Coef':>10}  {'Pip equiv':>10}  {'dWR pp':>10}")
print(f"  " + "-" * 48)
for i, col in enumerate(port_cols):
    c = lr2.coef_[0][i + 1]
    equiv = c / pip_coef2
    dwp   = c * p0 * (1 - p0) * 100
    print(f"  {port_labels[col]:>15}  {c:>+10.4f}  {equiv:>+9.1f}  {dwp:>+9.2f}pp")

# ── Direct binning validation: win rate by pip bucket × port status ────────────
print()
print("=" * 62)
print("Direct validation: win rate by pip bucket × 3:1 port access")
print("(pip bucket = total_pip at turn 10; each row is a pip range)")
print("-" * 62)

snap["pip_bucket"] = pd.cut(snap["total_pip"],
                             bins=[0, 12, 16, 20, 24, 28, 32, 36, 100],
                             labels=["0–12","12–16","16–20","20–24","24–28","28–32","32–36","36+"])

tbl = snap.groupby(["pip_bucket", "port_generic"], observed=True).agg(
    n=("is_winner", "count"),
    wr=("is_winner", "mean"),
).reset_index()
tbl["wr"] = (tbl["wr"] * 100).round(1)

print(f"  {'Pip range':>10}  {'No port (n)':>14}  {'WR':>6}  {'3:1 port (n)':>14}  {'WR':>6}  {'Port lift':>10}")
print(f"  " + "-" * 68)
for bucket in tbl["pip_bucket"].unique():
    sub = tbl[tbl["pip_bucket"] == bucket]
    row_no   = sub[sub["port_generic"] == 0]
    row_port = sub[sub["port_generic"] == 1]
    if row_no.empty or row_port.empty:
        continue
    n_no   = int(row_no["n"].values[0])
    wr_no  = row_no["wr"].values[0]
    n_yes  = int(row_port["n"].values[0])
    wr_yes = row_port["wr"].values[0]
    lift   = wr_yes - wr_no
    print(f"  {str(bucket):>10}  {n_no:>14,}  {wr_no:>5.1f}%  {n_yes:>14,}  {wr_yes:>5.1f}%  {lift:>+9.1f}pp")

print()
print("Reading: a player in the 24–28 pip bucket WITH a 3:1 port should")
print("have a similar win rate to a player in which bucket WITHOUT a port?")
print("That gap = the pip value of the port.\n")
