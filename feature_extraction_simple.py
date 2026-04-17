"""
Option A feature extraction: single forward pass through event logs.

State tracked cumulatively per game:
  - Settlements/cities from mapState.tileCornerStates deltas (buildingType 1/2)
  - Income from type-11 log events
  - Dev cards played/knights from type-20 log events
  - Times robbed from type-16 log events
  - Largest Army / Longest Road flags from playerStates.victoryPointsState deltas
  - Port access from corner (x,y,z) coordinate matching against portEdgeStates

Snapshots emitted every SNAPSHOT_INTERVAL completed turns.
One row per (game, player, snapshot_turn). Output: features.csv
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

SNAPSHOT_INTERVAL = 10
AVG_GAME_TURNS    = 70  # denominator for turn_frac (total_turns unknown at inference time)

data_dir   = Path("games/games")
game_files = list(data_dir.glob("*.json"))
print(f"Extracting features from {len(game_files):,} games...")

FIELDNAMES = [
    "game_id", "turn", "turn_frac",
    "player", "turn_order",
    "income", "income_share",
    "settlements", "cities",
    "public_vp", "vp_lead", "vp_share",
    "dev_played", "knights_played",
    "times_robbed",
    "la_flag", "lr_flag", "port_access",
    "total_turns",
    "rank", "is_winner",
]


def make_rows(turn, game_id, play_order, players,
              corner_btype, income, dev_played, knights, robbed,
              la_flag, lr_flag, port_access, total_turns, final_rank, winner_color):
    """Return a list of dicts (one per player) for a snapshot at `turn`."""
    per_player = []
    for p in players:
        s = sum(1 for (o, bt) in corner_btype.values() if o == p and bt == 1)
        c = sum(1 for (o, bt) in corner_btype.values() if o == p and bt == 2)
        vp = s + 2*c + la_flag[p]*2 + lr_flag[p]*2
        per_player.append((p, s, c, vp, income[p]))

    total_vp  = sum(r[3] for r in per_player) or 1
    total_inc = sum(r[4] for r in per_player) or 1

    rows = []
    for idx, (p, s, c, vp, inc) in enumerate(per_player):
        other_vp = [r[3] for i, r in enumerate(per_player) if i != idx]
        rows.append({
            "game_id":        game_id,
            "turn":           turn,
            "turn_frac":      round(turn / AVG_GAME_TURNS, 4),
            "player":         p,
            "turn_order":     play_order.index(p) + 1,
            "income":         inc,
            "income_share":   round(inc / total_inc, 4),
            "settlements":    s,
            "cities":         c,
            "public_vp":      vp,
            "vp_lead":        vp - max(other_vp) if other_vp else 0,
            "vp_share":       round(vp / total_vp, 4),
            "dev_played":     dev_played[p],
            "knights_played": knights[p],
            "times_robbed":   robbed[p],
            "la_flag":        la_flag[p],
            "lr_flag":        lr_flag[p],
            "port_access":    port_access[p],
            "total_turns":    total_turns,
            "rank":           final_rank.get(p, 4),
            "is_winner":      int(winner_color == p),
        })
    return rows


out_path   = Path("features.csv")
total_rows = 0
skipped    = 0

with open(out_path, "w", newline="") as out_f:
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    writer.writeheader()

    for gf in game_files:
        with open(gf) as f:
            game = json.load(f)

        data = game["data"]
        hist = data["eventHistory"]
        eg   = hist.get("endGameState", {})

        if not eg.get("players"):
            skipped += 1; continue

        play_order = data.get("playOrder", [])
        if len(play_order) != 4:
            skipped += 1; continue

        total_turns = eg.get("totalTurnCount", 0)
        if total_turns == 0:
            skipped += 1; continue

        players = [int(c) for c in play_order]

        final_rank   = {int(cs): p.get("rank", 4) for cs, p in eg["players"].items()}
        winner_color = next((int(cs) for cs, p in eg["players"].items()
                             if p.get("winningPlayer")), None)

        # Build port-corner lookup from initialState
        init_map    = hist.get("initialState", {}).get("mapState", {})
        port_coords = {
            (v.get("x"), v.get("y"), v.get("z"))
            for v in init_map.get("portEdgeStates", {}).values()
        }
        corner_xy = {
            cid: (c.get("x"), c.get("y"), c.get("z"))
            for cid, c in init_map.get("tileCornerStates", {}).items()
        }

        # Per-player cumulative state
        income      = defaultdict(int)
        dev_played  = defaultdict(int)
        knights     = defaultdict(int)
        robbed      = defaultdict(int)
        la_flag     = defaultdict(int)
        lr_flag     = defaultdict(int)
        port_access = defaultdict(int)
        corner_btype = {}   # cid -> (owner, buildingType)

        cur_turns          = 0
        last_snapshot_turn = -1

        for ev in hist["events"]:
            sc = ev.get("stateChange", {})

            # Turn counter — emit snapshots when crossing interval boundaries
            cs_state = sc.get("currentState", {})
            if "completedTurns" in cs_state:
                cur_turns = cs_state["completedTurns"]
                next_snap = last_snapshot_turn + SNAPSHOT_INTERVAL
                while next_snap <= cur_turns:
                    if next_snap > 0:
                        for row in make_rows(next_snap, gf.stem, play_order, players,
                                             corner_btype, income, dev_played, knights,
                                             robbed, la_flag, lr_flag, port_access,
                                             total_turns, final_rank, winner_color):
                            writer.writerow(row)
                            total_rows += 1
                    last_snapshot_turn = next_snap
                    next_snap += SNAPSHOT_INTERVAL

            # Map state: track buildings via tileCornerStates delta
            for cid, cdata in sc.get("mapState", {}).get("tileCornerStates", {}).items():
                if "owner" not in cdata:
                    continue
                owner = cdata["owner"]
                bt    = cdata.get("buildingType", 0)
                if bt not in (1, 2):
                    continue
                corner_btype[cid] = (owner, bt)
                if corner_xy.get(cid) in port_coords:
                    port_access[owner] = 1

            # Player states: LA / LR flag deltas
            for cs_str, pstate in sc.get("playerStates", {}).items():
                try:
                    color = int(cs_str)
                except ValueError:
                    continue
                vps = pstate.get("victoryPointsState", {})
                if "3" in vps:
                    la_flag[color] = int(vps["3"])
                if "4" in vps:
                    lr_flag[color] = int(vps["4"])

            # Game log events
            for log in sc.get("gameLogState", {}).values():
                t  = log.get("text", {})
                tt = t.get("type")
                pc = t.get("playerColor")

                if tt == 11 and pc:                       # resource income from roll
                    income[pc] += 1
                elif tt == 20 and pc:                     # dev card played
                    dev_played[pc] += 1
                    if t.get("cardEnum") in (10, 11):     # knight
                        knights[pc] += 1
                elif tt == 16:                            # steal
                    victim = t.get("playerColorVictim")
                    if victim:
                        robbed[victim] += 1

        # Final snapshot at game-end state
        if cur_turns != last_snapshot_turn and cur_turns > 0:
            for row in make_rows(cur_turns, gf.stem, play_order, players,
                                 corner_btype, income, dev_played, knights,
                                 robbed, la_flag, lr_flag, port_access,
                                 total_turns, final_rank, winner_color):
                writer.writerow(row)
                total_rows += 1

print(f"Done. {total_rows:,} rows written to {out_path}  ({skipped} games skipped)")
