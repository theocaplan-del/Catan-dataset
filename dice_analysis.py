import json
from pathlib import Path
from collections import defaultdict
import math

data_dir = Path("games/games")
game_files = list(data_dir.glob("*.json"))
print(f"Analyzing {len(game_files):,} games...\n")

dice_counts = defaultdict(int)
total_rolls = 0

for game_file in game_files:
    with open(game_file) as f:
        game = json.load(f)
    events = game["data"]["eventHistory"]["events"]
    for event in events:
        gls = event.get("stateChange", {}).get("gameLogState", {})
        for log in gls.values():
            text = log.get("text", {})
            if text.get("type") == 10:
                d1 = text.get("firstDice")
                d2 = text.get("secondDice")
                if d1 is not None and d2 is not None:
                    dice_counts[d1 + d2] += 1
                    total_rolls += 1

print(f"Total dice rolls: {total_rolls:,}\n")

# Expected probabilities for 2d6
expected_prob = {
    2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
    8: 5, 9: 4, 10: 3, 11: 2, 12: 1
}
total_ways = 36

print(f"{'Roll':>4}  {'Observed':>10}  {'Obs%':>7}  {'Exp%':>7}  {'Diff':>7}")
print("-" * 45)

chi_sq = 0.0
for roll in range(2, 13):
    observed = dice_counts[roll]
    obs_pct = observed / total_rolls * 100
    exp_frac = expected_prob[roll] / total_ways
    exp_pct = exp_frac * 100
    diff = obs_pct - exp_pct
    expected_count = total_rolls * exp_frac
    chi_sq += (observed - expected_count) ** 2 / expected_count
    flag = " <--" if abs(diff) > 0.3 else ""
    print(f"{roll:>4}  {observed:>10,}  {obs_pct:>7.3f}%  {exp_pct:>7.3f}%  {diff:>+7.3f}%{flag}")

print("-" * 45)
print(f"\nChi-squared statistic: {chi_sq:.4f}")
print(f"Degrees of freedom: 10")

# p-value approximation using chi-squared CDF (10 df)
# Critical values: p=0.05 -> 18.307, p=0.01 -> 23.209, p=0.001 -> 29.588
if chi_sq < 18.307:
    verdict = "PASS (p > 0.05) — no significant deviation from fair dice"
elif chi_sq < 23.209:
    verdict = "BORDERLINE (0.01 < p < 0.05) — marginal deviation"
elif chi_sq < 29.588:
    verdict = "FAIL (0.001 < p < 0.01) — significant deviation"
else:
    verdict = "FAIL (p < 0.001) — highly significant deviation"

print(f"Result: {verdict}\n")

# Also check individual dice fairness
d1_counts = defaultdict(int)
d2_counts = defaultdict(int)
for game_file in game_files:
    with open(game_file) as f:
        game = json.load(f)
    events = game["data"]["eventHistory"]["events"]
    for event in events:
        gls = event.get("stateChange", {}).get("gameLogState", {})
        for log in gls.values():
            text = log.get("text", {})
            if text.get("type") == 10:
                d1 = text.get("firstDice")
                d2 = text.get("secondDice")
                if d1 is not None and d2 is not None:
                    d1_counts[d1] += 1
                    d2_counts[d2] += 1

print("Individual die fairness (each face should be ~16.67%):")
print(f"{'Face':>4}  {'Die1%':>7}  {'Die2%':>7}")
print("-" * 22)
for face in range(1, 7):
    p1 = d1_counts[face] / total_rolls * 100
    p2 = d2_counts[face] / total_rolls * 100
    print(f"{face:>4}  {p1:>7.3f}%  {p2:>7.3f}%")
