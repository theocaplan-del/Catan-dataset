"""
Robber temporal and duration impact analysis.

A) Does the robber hurt more early game (higher share of production blocked)?
B) Does blocking more ACTUAL rolls → worse victim outcome?
C) Income-per-roll comparison: before / during / after blockage window.
   If the robber suppresses production, we expect:
     during < before,  after ≈ before  (or a rebound if pent-up demand)
D) Heuristic: WR cost per blocked roll → approximate value of placement.

Key definitions:
  - lost_rolls: number of times the blocked dice number was actually rolled
    while the robber sat on that tile (between this type-49 and the next).
  - income-per-roll (IPR): victim's type-11 income events divided by total
    dice rolls in the same window (normalises for variable window length).
  - Blockage window: [this type-49 + 1 .. next type-49).
  - Before/after windows: same event-count length, immediately adjacent.
"""

import json
from pathlib import Path
from collections import defaultdict

data_dir = Path("games/games")
game_files = list(data_dir.glob("*.json"))
print(f"Analyzing {len(game_files):,} games...\n")

records = []

for gf in game_files:
    with open(gf) as f:
        game = json.load(f)
    data = game["data"]
    events = data["eventHistory"]["events"]
    eg = data["eventHistory"].get("endGameState", {})
    total_turns = eg.get("totalTurnCount", 0)
    if total_turns == 0:
        continue

    final_rank = {int(cs): p.get("rank", 4) for cs, p in eg.get("players", {}).items()}

    # Flat annotated list: (completed_turns, logs)
    annotated = []
    cur_turns = 0
    for ev in events:
        sc = ev.get("stateChange", {})
        ct_state = sc.get("currentState", {})
        if "completedTurns" in ct_state:
            cur_turns = ct_state["completedTurns"]
        annotated.append((cur_turns, list(sc.get("gameLogState", {}).values())))

    # Collect all type-49 placements with thief + optional victim
    last_knight = None
    placements = []  # (ann_idx, completed_turns, rob_dn, thief, victim)

    for ann_idx, (ct, logs) in enumerate(annotated):
        types_in = {log.get("text", {}).get("type") for log in logs}

        for log in logs:
            t = log.get("text", {})
            if t.get("type") == 20 and t.get("cardEnum") in (10, 11):
                last_knight = t.get("playerColor")

        if 49 not in types_in:
            continue

        rob_dn = None
        for log in logs:
            t = log.get("text", {})
            if t.get("type") == 49:
                dn = t.get("tileInfo", {}).get("diceNumber")
                rt = t.get("tileInfo", {}).get("resourceType")
                if dn and rt:
                    rob_dn = dn
                break
        if rob_dn is None:
            continue

        thief = None
        if 10 in types_in:
            for log in logs:
                t = log.get("text", {})
                if t.get("type") == 10:
                    thief = t.get("playerColor")
                    break
        elif last_knight is not None:
            thief = last_knight
            last_knight = None

        victim = None
        for j in range(max(0, ann_idx - 2), min(len(annotated), ann_idx + 3)):
            for log in annotated[j][1]:
                t = log.get("text", {})
                if t.get("type") == 16:
                    v = t.get("playerColorVictim")
                    if v and v != thief:
                        victim = v
                        break
            if victim:
                break

        placements.append((ann_idx, ct, rob_dn, thief, victim))

    n = len(placements)

    def count_window(start, end, victim_color, rob_dn):
        rolls = 0; lost = 0; income = 0
        for j in range(start, end):
            for log in annotated[j][1]:
                t = log.get("text", {})
                tt = t.get("type")
                if tt == 10:
                    d1, d2 = t.get("firstDice"), t.get("secondDice")
                    if d1 is not None and d2 is not None:
                        rolls += 1
                        if d1 + d2 == rob_dn:
                            lost += 1
                elif tt == 11 and t.get("playerColor") == victim_color:
                    income += 1
        return rolls, lost, income

    for p_idx in range(n):
        ann_idx, ct, rob_dn, thief, victim = placements[p_idx]
        if thief is None or victim is None:
            continue

        end_idx  = placements[p_idx + 1][0] if p_idx + 1 < n else len(annotated)
        win_len  = end_idx - ann_idx - 1          # blockage window length (events)

        # During blockage
        rd, lr, id_ = count_window(ann_idx + 1, end_idx, victim, rob_dn)

        # Before blockage (same length, immediately preceding)
        b_start = max(0, ann_idx - win_len)
        rb, _,  ib = count_window(b_start, ann_idx, victim, rob_dn)

        # After blockage (same length, immediately following)
        a_end = min(len(annotated), end_idx + win_len)
        ra, _, ia = count_window(end_idx, a_end, victim, rob_dn)

        records.append({
            "game_progress": ct / total_turns,
            "rob_dn":        rob_dn,
            "lost_rolls":    lr,
            "rd": rd, "rb": rb, "ra": ra,   # dice rolls in each window
            "id": id_, "ib": ib, "ia": ia,  # victim income events in each window
            "victim_rank":   final_rank.get(victim, 4),
            "victim_won":    final_rank.get(victim, 4) == 1,
        })

print(f"Records (identified thief + victim): {len(records):,}\n")

def vwr(lst):
    return sum(x["victim_won"] for x in lst) / len(lst) * 100 if lst else 0

def avr(lst):
    return sum(x["victim_rank"] for x in lst) / len(lst) if lst else 0

def ipr(income_key, rolls_key, lst):
    """Mean income-per-roll across records, ignoring zero-roll windows."""
    vals = [r[income_key] / r[rolls_key] for r in lst if r[rolls_key] > 0]
    return sum(vals) / len(vals) if vals else float("nan")

# ── A) Game stage impact ───────────────────────────────────────────────────────
print("=" * 70)
print("A) Does the robber hurt the victim more EARLY in the game?")
print("   (random baseline victim WR = 25%)")
print("-" * 70)
quartiles = [[], [], [], []]
for r in records:
    q = min(int(r["game_progress"] * 4), 3)
    quartiles[q].append(r)
qlabels = ["Q1 early (0–25%)", "Q2 (25–50%)", "Q3 (50–75%)", "Q4 late (75–100%)"]
print(f"  {'Stage':>18}  {'n':>7}  {'Victim WR':>10}  {'Avg rank':>10}  {'Avg lost rolls':>15}")
print("  " + "-" * 66)
for q, grp in enumerate(quartiles):
    lr_avg = sum(x["lost_rolls"] for x in grp) / len(grp) if grp else 0
    print(f"  {qlabels[q]:>18}  {len(grp):>7,}  {vwr(grp):>9.1f}%  {avr(grp):>10.2f}  {lr_avg:>15.2f}")

# ── B) Blocked rolls vs victim outcome ────────────────────────────────────────
print()
print("=" * 70)
print("B) Actual blocked rolls vs victim outcome")
print("   (lost_rolls = times the blocked dice number was rolled while robber sat there)")
print("-" * 70)
from collections import defaultdict as DD
lr_groups = DD(list)
for r in records:
    lr_groups[min(r["lost_rolls"], 5)].append(r)
print(f"  {'Lost rolls':>12}  {'n':>7}  {'Victim WR':>10}  {'Avg rank':>10}")
print("  " + "-" * 44)
for lr in sorted(lr_groups):
    grp = lr_groups[lr]
    label = "5+" if lr == 5 else str(lr)
    print(f"  {label:>12}  {len(grp):>7,}  {vwr(grp):>9.1f}%  {avr(grp):>10.2f}")

# ── C) Income suppression: before / during / after ────────────────────────────
print()
print("=" * 70)
print("C) Income suppression (income-per-roll, symmetric windows)")
print("   before = window before placement; during = blockage; after = post-removal")
print("-" * 70)

valid3 = [r for r in records if r["rb"] > 0 and r["rd"] > 0 and r["ra"] > 0]
print(f"  Records with non-zero rolls in all 3 windows: {len(valid3):,}")
b_rate = ipr("ib", "rb", valid3)
d_rate = ipr("id", "rd", valid3)
a_rate = ipr("ia", "ra", valid3)
print(f"  Mean income/roll  BEFORE : {b_rate:.4f}")
print(f"  Mean income/roll  DURING : {d_rate:.4f}  ({(d_rate/b_rate-1)*100:+.1f}% vs before)")
print(f"  Mean income/roll  AFTER  : {a_rate:.4f}  ({(a_rate/b_rate-1)*100:+.1f}% vs before)")

print()
print(f"  Broken down by lost_rolls (using valid3 subset):")
print(f"  {'Lost':>6}  {'n':>7}  {'IPR before':>12}  {'IPR during':>12}  {'IPR after':>12}  {'Suppression':>12}")
print("  " + "-" * 68)
for lr in range(6):
    grp = [r for r in valid3 if min(r["lost_rolls"], 5) == lr]
    if len(grp) < 50:
        continue
    b = ipr("ib", "rb", grp)
    d = ipr("id", "rd", grp)
    a = ipr("ia", "ra", grp)
    sup = (1 - d/b) * 100 if b > 0 else 0
    label = "5+" if lr == 5 else str(lr)
    print(f"  {label:>6}  {len(grp):>7,}  {b:>12.4f}  {d:>12.4f}  {a:>12.4f}  {sup:>11.1f}%")

# Also by game stage
print()
print(f"  Broken down by game stage:")
print(f"  {'Stage':>18}  {'n':>7}  {'IPR before':>12}  {'IPR during':>12}  {'IPR after':>12}  {'Suppression':>12}")
print("  " + "-" * 80)
for q, grp_all in enumerate(quartiles):
    grp = [r for r in grp_all if r["rb"] > 0 and r["rd"] > 0 and r["ra"] > 0]
    if len(grp) < 50:
        continue
    b = ipr("ib", "rb", grp)
    d = ipr("id", "rd", grp)
    a = ipr("ia", "ra", grp)
    sup = (1 - d/b) * 100 if b > 0 else 0
    print(f"  {qlabels[q]:>18}  {len(grp):>7,}  {b:>12.4f}  {d:>12.4f}  {a:>12.4f}  {sup:>11.1f}%")

# ── D) Heuristic value estimate ───────────────────────────────────────────────
print()
print("=" * 70)
print("D) Heuristic: approximate value of one blocked roll")
print("-" * 70)
lr0  = lr_groups[0]
lr1  = lr_groups[1]
lr2  = lr_groups[2]
lr3p = [r for r in records if r["lost_rolls"] >= 3]

print(f"  Victim WR at 0 lost rolls : {vwr(lr0):>5.2f}%  (n={len(lr0):,})")
print(f"  Victim WR at 1 lost roll  : {vwr(lr1):>5.2f}%  (n={len(lr1):,})")
print(f"  Victim WR at 2 lost rolls : {vwr(lr2):>5.2f}%  (n={len(lr2):,})")
print(f"  Victim WR at 3+ lost rolls: {vwr(lr3p):>5.2f}%  (n={len(lr3p):,})")
print()

d01 = vwr(lr1) - vwr(lr0)
d12 = vwr(lr2) - vwr(lr1)
print(f"  Marginal WR cost to victim per blocked roll:")
print(f"    0 → 1 : {d01:+.2f}pp")
print(f"    1 → 2 : {d12:+.2f}pp")
print()
print(f"  Probability the blocked number is rolled on any given turn:")
print(f"    6 or 8  tile (5 pips): {5/36:.4f}  → expected 1 blocked roll per {36/5:.1f} turns")
print(f"    5 or 9  tile (4 pips): {4/36:.4f}  → expected 1 blocked roll per {36/4:.1f} turns")
print(f"    4 or 10 tile (3 pips): {3/36:.4f}  → expected 1 blocked roll per {36/3:.1f} turns")
print()
if d01 < 0:
    print(f"  Estimated WR cost per TURN of blockage (using 0→1 marginal):")
    for label, pip in [("6/8  (5 pip)", 5), ("5/9  (4 pip)", 4), ("4/10 (3 pip)", 3)]:
        cost_per_turn = d01 * (pip / 36)
        print(f"    {label}: {cost_per_turn:+.4f} pp/turn")
    print()
    print(f"  Estimated WR cost for a 10-turn blockage:")
    for label, pip in [("6/8  (5 pip)", 5), ("5/9  (4 pip)", 4), ("4/10 (3 pip)", 3)]:
        exp_lost = 10 * pip / 36
        cost = d01 * exp_lost
        print(f"    {label}: expected {exp_lost:.1f} lost rolls → {cost:+.2f} pp")
