"""
Win-rate prediction model (Option A).

Trains a LightGBM binary classifier on per-player turn snapshots.
Cross-validation is game-grouped (no game appears in both train and val).
After CV, softmax-calibrates 4-player predictions per snapshot so they sum to 1.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("features.csv")
print(f"Loaded {len(df):,} rows from {df['game_id'].nunique():,} games\n")

# Filter: only turns with some meaningful game state (skip very early snapshots)
df = df[df["turn"] >= 10].copy()
print(f"After filtering turn >= 10: {len(df):,} rows\n")

FEATURES = [
    "turn", "turn_frac", "turn_order",
    "income", "income_share",
    "settlements", "cities",
    "public_vp", "vp_lead", "vp_share",
    "dev_played", "knights_played",
    "times_robbed",
    "la_flag", "lr_flag", "port_access",
]

X      = df[FEATURES].values
y      = df["is_winner"].values
groups = df["game_id"].values

# ── 5-fold game-grouped cross-validation ─────────────────────────────────────
print("Running 5-fold group CV (grouped by game)...")
gkf         = GroupKFold(n_splits=5)
oof_preds   = np.zeros(len(df))
auc_scores  = []
brier_scores = []
last_model  = None

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = preds
    last_model = model

    auc   = roc_auc_score(y_val, preds)
    brier = brier_score_loss(y_val, preds)
    auc_scores.append(auc)
    brier_scores.append(brier)
    print(f"  Fold {fold + 1}: AUC={auc:.4f}  Brier={brier:.4f}  "
          f"best_iter={model.best_iteration_}")

print(f"\nOOF AUC:   {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"OOF Brier: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")

# ── Softmax calibration across 4 players per snapshot ─────────────────────────
df["raw_pred"] = oof_preds
df["win_prob"] = 0.0

for (gid, turn), grp in df.groupby(["game_id", "turn"]):
    raw   = grp["raw_pred"].values
    probs = np.exp(raw - raw.max())        # numerically stable softmax
    probs /= probs.sum()
    df.loc[grp.index, "win_prob"] = probs

auc_cal   = roc_auc_score(df["is_winner"], df["win_prob"])
brier_cal = brier_score_loss(df["is_winner"], df["win_prob"])
print(f"\nAfter softmax calibration:")
print(f"  AUC:   {auc_cal:.4f}")
print(f"  Brier: {brier_cal:.4f}")
print(f"  Mean predicted win prob: {df['win_prob'].mean()*100:.1f}%  (target = 25.0%)")

# ── Feature importances ────────────────────────────────────────────────────────
print(f"\nFeature importances (from last fold):")
imp = pd.Series(last_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, v in imp.items():
    bar = "█" * (v * 40 // imp.max())
    print(f"  {feat:<22} {v:>5}  {bar}")

# ── AUC by game stage ─────────────────────────────────────────────────────────
print(f"\nAUC by game stage (turn_frac buckets):")
df["stage"] = pd.cut(df["turn_frac"], bins=[0, 0.25, 0.5, 0.75, 1.0, 99],
                     labels=["0–25%", "25–50%", "50–75%", "75–100%", ">100%"])
for stage, grp in df.groupby("stage", observed=True):
    if grp["is_winner"].sum() < 10:
        continue
    auc_s = roc_auc_score(grp["is_winner"], grp["win_prob"])
    n_snaps = len(grp) // 4
    print(f"  {stage}: {n_snaps:>7,} snapshots  AUC={auc_s:.4f}")

# ── Calibration check: predicted prob deciles vs actual win rate ───────────────
print(f"\nCalibration (predicted win prob deciles vs actual win rate):")
df["decile"] = pd.qcut(df["win_prob"], q=10, labels=False, duplicates="drop")
cal = df.groupby("decile").agg(
    mean_pred=("win_prob", "mean"),
    actual_wr=("is_winner", "mean"),
    n=("is_winner", "count"),
).reset_index()
print(f"  {'Pred prob':>10}  {'Actual WR':>10}  {'n':>8}")
print(f"  " + "-" * 34)
for _, row in cal.iterrows():
    print(f"  {row['mean_pred']*100:>9.1f}%  {row['actual_wr']*100:>9.1f}%  {int(row['n']):>8,}")

# ── Sanity: early-game predictions should be near 25% ─────────────────────────
early = df[df["turn"] == 10]
print(f"\nAt turn 10:")
print(f"  Mean win_prob = {early['win_prob'].mean()*100:.1f}%  (expected ~25%)")
print(f"  Std  win_prob = {early['win_prob'].std()*100:.1f}%")
print(f"  Min/Max       = {early['win_prob'].min()*100:.1f}% / {early['win_prob'].max()*100:.1f}%")
