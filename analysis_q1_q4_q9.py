"""
Analysis of 5 questions using endGameState fields only (no full event parse).

VP key mapping (empirically verified - README had keys 2/3/4 wrong):
  "0" = settlement count       (1 VP each)
  "1" = city count             (2 VP each)
  "2" = dev VP card count      (1 VP each)  [hidden until win, not in type-20 events]
  "3" = largest army flag      (2 VP if 1)  [achievementEnum 1]
  "4" = longest road flag      (2 VP if 1)  [achievementEnum 0]
"""

import json
from pathlib import Path
from collections import defaultdict

data_dir = Path("games/games")
game_files = list(data_dir.glob("*.json"))
print(f"Loading {len(game_files):,} games...\n")

games = []
for gf in game_files:
    with open(gf) as f:
        g = json.load(f)
    data = g["data"]
    eg = data["eventHistory"].get("endGameState", {})
    if not eg.get("players"):
        continue

    play_order = data.get("playOrder", [])
    player_meta = {
        p.get("selectedColor"): p.get("isBot", False)
        for p in data.get("playerUserStates", [])
    }
    is_ranked = data.get("gameDetails", {}).get("isRanked", False)

    winner_color = None
    players = []
    for cs, p in eg["players"].items():
        color = int(cs)
        vp = p.get("victoryPoints", {})
        act = eg.get("activityStats", {}).get(cs, {})
        rs = eg.get("resourceStats", {}).get(cs, {})
        is_winner = bool(p.get("winningPlayer"))
        if is_winner:
            winner_color = color

        s = vp.get("0", 0)
        c = vp.get("1", 0)
        dv = vp.get("2", 0)
        la = vp.get("3", 0)   # achievementEnum 1
        lr = vp.get("4", 0)   # achievementEnum 0
        total_vp = s + c * 2 + dv + la * 2 + lr * 2

        players.append({
            "color": color,
            "is_winner": is_winner,
            "rank": p.get("rank", 0),
            "vp": {"s": s, "c": c, "dv": dv, "lr": lr, "la": la, "total": total_vp},
            "blocked": act.get("resourceIncomeBlocked", 0),
            "rolling_income": rs.get("rollingIncome", 0),
            "dev_bought": act.get("devCardsBought", 0),
            "dev_used": act.get("devCardsUsed", 0),
            "proposed_trades": act.get("proposedTrades", 0),
            "successful_trades": act.get("successfulTrades", 0),
            "is_bot": player_meta.get(color, False),
            "is_ranked": is_ranked,
        })

    if winner_color is not None:
        games.append({"players": players, "winner_color": winner_color})

print(f"Valid games: {len(games):,}\n")
print("=" * 60)

# ── Q1: VP Source Profiles ────────────────────────────────────
print("\nQ1: HOW DO WINNERS WIN? VP SOURCE PROFILES")
print("-" * 60)

all_winners = [p for g in games for p in g["players"] if p["is_winner"]]

vp_sources = defaultdict(int)
archetype_counts = defaultdict(int)
archetype_vp = defaultdict(lambda: defaultdict(float))

for w in all_winners:
    vp = w["vp"]
    s_vp  = vp["s"]
    c_vp  = vp["c"] * 2
    dv_vp = vp["dv"]
    la_vp = vp["la"] * 2
    lr_vp = vp["lr"] * 2

    vp_sources["settlements"] += s_vp
    vp_sources["cities"]      += c_vp
    vp_sources["dev_vp_cards"] += dv_vp
    vp_sources["largest_army"] += la_vp
    vp_sources["longest_road"] += lr_vp

    # Dominant archetype: what's the largest single VP source?
    sources = {
        "City Builder":      c_vp,
        "Settlement Spread": s_vp,
        "Achiever (LA+LR)":  la_vp + lr_vp,
        "Dev Card VP":       dv_vp,
    }
    dominant = max(sources, key=sources.get)
    archetype_counts[dominant] += 1
    for k, v in sources.items():
        archetype_vp[dominant][k] += v

total_winners = len(all_winners)
total_vp_distributed = sum(vp_sources.values())

print(f"\nVP source breakdown across {total_winners:,} winners:")
print(f"{'Source':<20} {'Total VP':>10}  {'% of VP':>8}  {'Avg per winner':>14}")
print("-" * 58)
for src, tvp in sorted(vp_sources.items(), key=lambda x: -x[1]):
    print(f"{src:<20} {tvp:>10,}  {tvp/total_vp_distributed*100:>7.1f}%  {tvp/total_winners:>14.2f}")

print(f"\nWinner archetypes (dominant VP source):")
print(f"{'Archetype':<24} {'Count':>8}  {'%':>6}")
print("-" * 42)
for arch, cnt in sorted(archetype_counts.items(), key=lambda x: -x[1]):
    print(f"{arch:<24} {cnt:>8,}  {cnt/total_winners*100:>5.1f}%")

# Fraction with each bonus
has_la = sum(1 for w in all_winners if w["vp"]["la"] > 0)  # key "3"
has_lr = sum(1 for w in all_winners if w["vp"]["lr"] > 0)  # key "4"
has_dv = sum(1 for w in all_winners if w["vp"]["dv"] > 0)  # key "2"
print(f"\nBonus holdings among winners:")
print(f"  Has Largest Army:  {has_la:,} / {total_winners:,} ({has_la/total_winners*100:.1f}%)")
print(f"  Has Longest Road:  {has_lr:,} / {total_winners:,} ({has_lr/total_winners*100:.1f}%)")
print(f"  Has Dev VP cards:  {has_dv:,} / {total_winners:,} ({has_dv/total_winners*100:.1f}%)")

print("\n" + "=" * 60)

# ── Q2: Largest Army vs Longest Road ─────────────────────────
print("\nQ2: IS LARGEST ARMY OR LONGEST ROAD MORE VALUABLE?")
print("-" * 60)

la_games = lr_games = both_games = neither_games = 0
la_wins = lr_wins = both_wins = neither_wins = 0

for game in games:
    for p in game["players"]:
        has_la = p["vp"]["la"] > 0
        has_lr = p["vp"]["lr"] > 0
        w = p["is_winner"]
        if has_la and has_lr:
            both_games += 1
            if w: both_wins += 1
        elif has_la:
            la_games += 1
            if w: la_wins += 1
        elif has_lr:
            lr_games += 1
            if w: lr_wins += 1
        else:
            neither_games += 1
            if w: neither_wins += 1

def wr(wins, total):
    return wins / total * 100 if total else 0

print(f"\n{'Status':<22} {'Players':>10}  {'Wins':>8}  {'Win Rate':>10}")
print("-" * 54)
print(f"{'LA only':<22} {la_games:>10,}  {la_wins:>8,}  {wr(la_wins,la_games):>9.1f}%")
print(f"{'LR only':<22} {lr_games:>10,}  {lr_wins:>8,}  {wr(lr_wins,lr_games):>9.1f}%")
print(f"{'Both LA + LR':<22} {both_games:>10,}  {both_wins:>8,}  {wr(both_wins,both_games):>9.1f}%")
print(f"{'Neither':<22} {neither_games:>10,}  {neither_wins:>8,}  {wr(neither_wins,neither_games):>9.1f}%")

print(f"\nBaseline (random): {1/4*100:.1f}%")
print(f"LA lift over baseline: {wr(la_wins,la_games) - 25:.1f}pp")
print(f"LR lift over baseline: {wr(lr_wins,lr_games) - 25:.1f}pp")

print("\n" + "=" * 60)

# ── Q3: Does getting robbed more = winning? ───────────────────
print("\nQ3: DOES GETTING ROBBED MORE MEAN YOU'RE WINNING?")
print("-" * 60)

all_players = [p for g in games for p in g["players"]]

rank_blocked = defaultdict(list)
for p in all_players:
    rank_blocked[p["rank"]].append(p["blocked"])

print(f"\n{'Rank':<8} {'Mean blocked':>14}  {'Median':>8}  {'Players':>10}")
print("-" * 46)
for rank in sorted(rank_blocked):
    vals = rank_blocked[rank]
    vals_sorted = sorted(vals)
    median = vals_sorted[len(vals_sorted) // 2]
    print(f"{rank:<8} {sum(vals)/len(vals):>14.2f}  {median:>8}  {len(vals):>10,}")

# Simple correlation: rank vs blocked (1=winner, so inverse = more blocked = lower rank?)
import statistics
blocked_winners = [p["blocked"] for p in all_players if p["rank"] == 1]
blocked_losers  = [p["blocked"] for p in all_players if p["rank"] != 1]
print(f"\nWinners avg blocked:    {sum(blocked_winners)/len(blocked_winners):.2f}")
print(f"Non-winners avg blocked: {sum(blocked_losers)/len(blocked_losers):.2f}")
print(f"\nConclusion: {'Winners are robbed MORE' if sum(blocked_winners)/len(blocked_winners) > sum(blocked_losers)/len(blocked_losers) else 'Winners are robbed LESS'} than non-winners")

print("\n" + "=" * 60)

# ── Q4: Optimal resource income ──────────────────────────────
print("\nQ4: WHAT'S THE OPTIMAL ROLLING INCOME FOR A WINNER?")
print("-" * 60)

from collections import defaultdict as DD

bucket_wins = DD(int)
bucket_total = DD(int)
bucket_income = DD(list)

for p in all_players:
    inc = p["rolling_income"]
    bucket = (inc // 10) * 10
    bucket = min(bucket, 90)  # cap at 90+
    bucket_total[bucket] += 1
    if p["is_winner"]:
        bucket_wins[bucket] += 1
    bucket_income[p["rank"]].append(inc)

print(f"\nRolling income buckets vs win rate:")
print(f"{'Income range':>15}  {'Players':>10}  {'Wins':>8}  {'Win Rate':>10}")
print("-" * 50)
for b in sorted(set(bucket_total)):
    label = f"{b}-{b+9}" if b < 90 else "90+"
    tot = bucket_total[b]
    wins = bucket_wins[b]
    print(f"{label:>15}  {tot:>10,}  {wins:>8,}  {wr(wins,tot):>9.1f}%")

print(f"\nAverage rolling income by final rank:")
for rank in sorted(bucket_income):
    vals = bucket_income[rank]
    print(f"  Rank {rank}: mean={sum(vals)/len(vals):.1f}, median={sorted(vals)[len(vals)//2]}")

print("\n" + "=" * 60)

# ── Q9: Bots vs Humans ────────────────────────────────────────
print("\nQ9: DO BOTS PLAY DIFFERENTLY FROM HUMANS?")
print("-" * 60)

bot_players   = [p for p in all_players if p["is_bot"]]
human_players = [p for p in all_players if not p["is_bot"]]

def avg(lst, key):
    vals = [x[key] for x in lst if x[key] is not None]
    return sum(vals) / len(vals) if vals else 0

def win_rate(lst):
    return sum(1 for p in lst if p["is_winner"]) / len(lst) * 100 if lst else 0

print(f"\nBot players:   {len(bot_players):,}")
print(f"Human players: {len(human_players):,}")
print()
print(f"{'Metric':<25} {'Bots':>10}  {'Humans':>10}  {'Diff':>10}")
print("-" * 60)
metrics = [
    ("Win rate (%)",       lambda lst: win_rate(lst)),
    ("Avg rolling income", lambda lst: avg(lst, "rolling_income")),
    ("Avg dev cards bought",lambda lst: avg(lst, "dev_bought")),
    ("Avg dev cards used",  lambda lst: avg(lst, "dev_used")),
    ("Avg proposed trades", lambda lst: avg(lst, "proposed_trades")),
    ("Avg successful trades",lambda lst: avg(lst, "successful_trades")),
    ("Avg times blocked",   lambda lst: avg(lst, "blocked")),
]
for label, fn in metrics:
    bv = fn(bot_players)
    hv = fn(human_players)
    diff = bv - hv
    print(f"{label:<25} {bv:>10.2f}  {hv:>10.2f}  {diff:>+10.2f}")

# Win rate by rank for bots vs humans
print(f"\nBot vs Human win rate by game composition:")
games_with_bots    = [g for g in games if any(p["is_bot"] for p in g["players"])]
games_without_bots = [g for g in games if not any(p["is_bot"] for p in g["players"])]
print(f"  Games with ≥1 bot:    {len(games_with_bots):,}")
print(f"  Games with 0 bots:    {len(games_without_bots):,}")

if games_with_bots:
    bot_wr_in_mixed   = sum(1 for g in games_with_bots for p in g["players"] if p["is_bot"] and p["is_winner"])
    bot_total_in_mixed = sum(1 for g in games_with_bots for p in g["players"] if p["is_bot"])
    human_wr_in_mixed  = sum(1 for g in games_with_bots for p in g["players"] if not p["is_bot"] and p["is_winner"])
    human_total_in_mixed = sum(1 for g in games_with_bots for p in g["players"] if not p["is_bot"])
    if bot_total_in_mixed and human_total_in_mixed:
        print(f"  Bots win rate (mixed games):   {bot_wr_in_mixed/bot_total_in_mixed*100:.1f}%")
        print(f"  Humans win rate (mixed games): {human_wr_in_mixed/human_total_in_mixed*100:.1f}%")
