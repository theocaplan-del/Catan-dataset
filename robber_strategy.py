"""
Robber strategy analysis: does targeting high-pip tiles vs scarce resources matter?

Approach:
  - type-49 gives the robbed TILE (diceNumber, resourceType)
  - The THIEF is identifiable from the same event:
      7-roll: type-10 playerColor in same event as type-49
      knight: type-20 playerColor in same or immediately preceding event as type-49
  - We aggregate per thief-game: avg pip of tiles they rob, victim's resource profile
  - For scarcity: we approximate the victim's resource reliance from their type-11 income
    over the game, then check whether the robbed resource is abundant or scarce for them.
    The VICTIM is from the nearest type-16 within 2 events.
"""

import json
from pathlib import Path
from collections import defaultdict

PIP = {2:1, 3:2, 4:3, 5:4, 6:5, 7:0, 8:5, 9:4, 10:3, 11:2, 12:1}
RESOURCES = {1:'Brick', 2:'Wool', 3:'Grain', 4:'Ore', 5:'Lumber'}

data_dir = Path("games/games")
game_files = list(data_dir.glob("*.json"))
print(f"Analyzing {len(game_files):,} games...\n")

# Per-placement record (one per type-49 event where we can identify the thief)
placements = []          # {pip, rt, thief_won, victim_frac}  -- only when victim known
pip_only = []            # {pip, thief_won}                    -- all placements

for gf in game_files:
    with open(gf) as f:
        game = json.load(f)
    data = game["data"]
    events = data["eventHistory"]["events"]
    eg = data["eventHistory"].get("endGameState", {})

    final_rank = {int(cs): p.get("rank", 4) for cs, p in eg.get("players", {}).items()}

    # Build each player's type-11 income by resource type (whole-game profile)
    player_resource_income = defaultdict(lambda: defaultdict(int))
    for ev in events:
        for log in ev.get("stateChange", {}).get("gameLogState", {}).values():
            t = log.get("text", {})
            if t.get("type") == 11:
                color = t.get("playerColor")
                rtype = t.get("tileInfo", {}).get("resourceType")
                if color and rtype:
                    player_resource_income[color][rtype] += 1

    # Walk events: find type-49 and identify the thief
    last_knight_player = None   # type-20 playerColor from very recent event

    for i, ev in enumerate(events):
        gls = ev.get("stateChange", {}).get("gameLogState", {})
        logs = list(gls.values())
        types_in = {v.get("text", {}).get("type") for v in logs}

        # Track knight player
        for log in logs:
            t = log.get("text", {})
            if t.get("type") == 20 and t.get("cardEnum") in (10, 11):
                last_knight_player = t.get("playerColor")

        if 49 not in types_in:
            continue

        # Get robber tile
        rob_tile = None
        for log in logs:
            t = log.get("text", {})
            if t.get("type") == 49:
                ti = t.get("tileInfo", {})
                dn, rt = ti.get("diceNumber"), ti.get("resourceType")
                if dn and rt:
                    rob_tile = (dn, rt)
                break
        if not rob_tile:
            continue

        rob_dn, rob_rt = rob_tile
        tile_pips = PIP.get(rob_dn, 0)

        # Identify thief
        thief = None
        if 10 in types_in:          # 7-roll: thief rolled the dice
            for log in logs:
                t = log.get("text", {})
                if t.get("type") == 10:
                    thief = t.get("playerColor")
                    break
        elif last_knight_player is not None:  # knight: track recent knight player
            thief = last_knight_player
            last_knight_player = None  # consume

        if thief is None:
            continue

        thief_won = (final_rank.get(thief, 4) == 1)
        pip_only.append({"pip": tile_pips, "rt": rob_rt, "thief_won": thief_won})

        # Find victim: look for type-16 within ±2 events
        victim = None
        for j in range(max(0, i-2), min(len(events), i+3)):
            gls2 = events[j].get("stateChange", {}).get("gameLogState", {})
            for log in gls2.values():
                t = log.get("text", {})
                if t.get("type") == 16:
                    v = t.get("playerColorVictim")
                    if v and v != thief:
                        victim = v
                        break
            if victim:
                break

        if victim is None:
            continue

        # Compute victim's reliance on robbed resource
        victim_income = player_resource_income[victim]
        total = sum(victim_income.values())
        if total == 0:
            continue
        victim_frac = victim_income.get(rob_rt, 0) / total

        placements.append({
            "pip":         tile_pips,
            "rt":          rob_rt,
            "victim_frac": victim_frac,
            "thief_won":   thief_won,
        })

print(f"Robber placements with identified thief: {len(pip_only):,}")
print(f"Placements with identified victim too:   {len(placements):,}\n")

def wr(lst, key="thief_won"):
    wins = sum(x[key] for x in lst)
    return wins / len(lst) * 100 if lst else 0

print("=" * 62)
print("Q: Does targeting high-pip tiles improve the thief's win rate?")
print("-" * 62)
print(f"{'Pip':>5}  {'Dice':>8}  {'Placements':>12}  {'Thief WR':>10}")
print("-" * 42)

from collections import defaultdict as DD
pip_groups = DD(list)
for p in pip_only:
    pip_groups[p["pip"]].append(p)

pip_labels = {0:"7(desert)", 1:"2,12", 2:"3,11", 3:"4,10", 4:"5,9", 5:"6,8"}
for pip in sorted(pip_groups):
    grp = pip_groups[pip]
    print(f"{pip:>5}  {pip_labels[pip]:>8}  {len(grp):>12,}  {wr(grp):>9.1f}%")

# ── Observed vs expected pip distribution ─────────────────────────────────────
print()
total_p = sum(len(v) for v in pip_groups.values() if 0 not in [x["pip"] for x in v])
total_all = len(pip_only)
expected = {1:2/36, 2:4/36, 3:6/36, 4:8/36, 5:10/36}
print("Observed vs expected % of placements (excl. desert):")
non_desert = [p for p in pip_only if p["pip"] > 0]
nd_total = len(non_desert)
print(f"{'Pip':>5}  {'Observed%':>12}  {'Expected%':>12}  {'Ratio':>8}")
pip_only_nod = DD(list)
for p in non_desert:
    pip_only_nod[p["pip"]].append(p)
for pip in sorted(expected):
    obs = len(pip_only_nod[pip]) / nd_total * 100
    exp = expected[pip] / sum(expected.values()) * 100
    ratio = obs / exp if exp > 0 else 0
    print(f"{pip:>5}  {obs:>11.1f}%  {exp:>11.1f}%  {ratio:>7.2f}x")

# ── Scarcity vs abundance targeting ───────────────────────────────────────────
print()
print("=" * 62)
print("Q: Does targeting the victim's scarce vs abundant resource matter?")
print("-" * 62)
print(f"{'Victim reliance':>22}  {'Placements':>12}  {'Thief WR':>10}")
print("-" * 48)

frac_groups = [[] for _ in range(5)]
for p in placements:
    b = min(int(p["victim_frac"] * 5), 4)
    frac_groups[b].append(p)

labels = ["0–20% (victim scarce)", "20–40%", "40–60%", "60–80%", "80–100% (abundant)"]
for b, grp in enumerate(frac_groups):
    print(f"{labels[b]:>22}  {len(grp):>12,}  {wr(grp):>9.1f}%")

# ── Resource type breakdown ────────────────────────────────────────────────────
print()
print("=" * 62)
print("Robber placement by resource type (% vs board frequency expected ~20%)")
print("-" * 62)
rt_groups = DD(list)
for p in pip_only:
    rt_groups[p["rt"]].append(p)
total_rt = len(pip_only)
for rt in sorted(rt_groups, key=lambda x: -len(rt_groups[x])):
    grp = rt_groups[rt]
    print(f"  {RESOURCES.get(rt,'?'):>8}: {len(grp):>7,}  ({len(grp)/total_rt*100:.1f}%)  thief WR {wr(grp):.1f}%")
