"""
Option B feature extraction: full board state reconstruction.

Adds over Option A:
  - Per-resource pip scores from initial board position
    (pip_brick, pip_wool, pip_grain, pip_ore, pip_lumber, total_pip)
  - Resource diversity: unique resources with any pip exposure
  - Per-resource income counts from type-11 events
  - Specific port types (port_generic, port_brick/wool/grain/ore/lumber)
  - Hand size from playerStates.resourceCards delta
  - Dev cards in hand from mechanicDevelopmentCardsState delta
  - Road count from tileEdgeStates delta
  - Robber-on-my-tile flag from mechanicRobberState

Corner→hex adjacency: flat-top hex grid, z=0=bottom-right going clockwise.
Empirically verified: for corner (q,r,z), adjacent hexes are at offsets CORNER_HEX_OFFSETS[z]
relative to (q,r). Multiple settlement verification confirms this is correct.

Hex tile type and type-11 resourceType use the same numbering:
  1=Brick  2=Wool  3=Grain  4=Ore  5=Lumber  0=Desert
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

SNAPSHOT_INTERVAL = 10
AVG_GAME_TURNS    = 70

PIP = {2:1, 3:2, 4:3, 5:4, 6:5, 7:0, 8:5, 9:4, 10:3, 11:2, 12:1}

# For corner (q, r, z), the 3 adjacent hexes are at these (dq, dr) offsets.
# z=0 bottom-right, going clockwise: 0=BR, 1=B, 2=BL, 3=TL, 4=T, 5=TR
CORNER_HEX_OFFSETS = {
    0: [(0, 0), (+1, 0), (0, +1)],
    1: [(0, 0), (0, +1), (-1, +1)],
    2: [(0, 0), (-1, +1), (-1, 0)],
    3: [(0, 0), (-1, 0), (0, -1)],
    4: [(0, 0), (0, -1), (+1, -1)],
    5: [(0, 0), (+1, -1), (+1, 0)],
}

RESOURCES   = [1, 2, 3, 4, 5]   # brick, wool, grain, ore, lumber
RT_NAMES    = {1:'brick', 2:'wool', 3:'grain', 4:'ore', 5:'lumber'}
# port type → pretty name (type 1 = generic/3:1)
PORT_TYPES  = {1:'generic', 2:'brick', 3:'grain', 4:'wool', 5:'lumber', 6:'ore'}

data_dir   = Path("games/games")
game_files = list(data_dir.glob("*.json"))
print(f"Extracting full-board features from {len(game_files):,} games...")

FIELDNAMES = [
    "game_id", "turn", "turn_frac",
    "player", "turn_order",
    # Board position
    "pip_brick", "pip_wool", "pip_grain", "pip_ore", "pip_lumber",
    "total_pip", "resource_diversity",
    # Per-resource income (actual)
    "income", "income_share",
    "income_brick", "income_wool", "income_grain", "income_ore", "income_lumber",
    # Buildings / VP
    "settlements", "cities",
    "public_vp", "vp_lead", "vp_share",
    # Cards / dev
    "dev_played", "knights_played", "dev_in_hand",
    # Hand
    "hand_size",
    # Roads
    "road_count",
    # Robber
    "times_robbed", "robber_on_my_tile",
    # Achievements
    "la_flag", "lr_flag",
    # Ports (specific types)
    "port_generic", "port_brick", "port_wool", "port_grain", "port_ore", "port_lumber",
    # Meta
    "total_turns", "rank", "is_winner",
]


def compute_pip_by_rt(player_corners, corner_xy, hex_lookup):
    """Per-resource pip score for a single player.
    player_corners: {cid: buildingType}  (only this player's buildings)
    Returns dict {resourceType: pip_score}
    """
    pip = defaultdict(float)
    for cid, bt in player_corners.items():
        mult = 2.0 if bt == 2 else 1.0
        xyz  = corner_xy.get(cid)
        if xyz is None:
            continue
        cx, cy, cz = xyz
        for dq, dr in CORNER_HEX_OFFSETS.get(cz, []):
            hdata = hex_lookup.get((cx + dq, cy + dr))
            if hdata:
                dn, rt = hdata
                if rt and dn:          # skip desert (rt=0) and off-board (None)
                    pip[rt] += PIP.get(dn, 0) * mult
    return pip


def make_rows(turn, game_id, play_order, players,
              all_player_corners, income, income_by_rt,
              dev_played, knights, robbed, dev_in_hand, hand_size,
              road_count, la_flag, lr_flag, player_ports,
              corner_xy, hex_lookup, robber_pos,
              total_turns, final_rank, winner_color):
    rows = []

    per_player = []
    for p in players:
        pc   = all_player_corners[p]
        s    = sum(1 for bt in pc.values() if bt == 1)
        c    = sum(1 for bt in pc.values() if bt == 2)
        vp   = s + 2*c + la_flag[p]*2 + lr_flag[p]*2
        inc  = income[p]
        pip  = compute_pip_by_rt(pc, corner_xy, hex_lookup)
        per_player.append((p, s, c, vp, inc, pip))

    total_vp  = sum(r[3] for r in per_player) or 1
    total_inc = sum(r[4] for r in per_player) or 1

    for idx, (p, s, c, vp, inc, pip) in enumerate(per_player):
        other_vp = [r[3] for i, r in enumerate(per_player) if i != idx]

        # Robber-on-my-tile: check if any of player's corners are adjacent to robber hex
        on_rob = 0
        if robber_pos:
            for cid in all_player_corners[p]:
                xyz = corner_xy.get(cid)
                if xyz:
                    cx, cy, cz = xyz
                    for dq, dr in CORNER_HEX_OFFSETS.get(cz, []):
                        if (cx + dq, cy + dr) == robber_pos:
                            on_rob = 1
                            break
                if on_rob:
                    break

        total_pip   = sum(pip.values())
        diversity   = sum(1 for rt in RESOURCES if pip.get(rt, 0) > 0)
        pp          = player_ports[p]

        rows.append({
            "game_id":        game_id,
            "turn":           turn,
            "turn_frac":      round(turn / AVG_GAME_TURNS, 4),
            "player":         p,
            "turn_order":     play_order.index(p) + 1,
            # Board position
            "pip_brick":      pip.get(1, 0),
            "pip_wool":       pip.get(2, 0),
            "pip_grain":      pip.get(3, 0),
            "pip_ore":        pip.get(4, 0),
            "pip_lumber":     pip.get(5, 0),
            "total_pip":      total_pip,
            "resource_diversity": diversity,
            # Income
            "income":         inc,
            "income_share":   round(inc / total_inc, 4),
            "income_brick":   income_by_rt[p][1],
            "income_wool":    income_by_rt[p][2],
            "income_grain":   income_by_rt[p][3],
            "income_ore":     income_by_rt[p][4],
            "income_lumber":  income_by_rt[p][5],
            # Buildings / VP
            "settlements":    s,
            "cities":         c,
            "public_vp":      vp,
            "vp_lead":        vp - max(other_vp) if other_vp else 0,
            "vp_share":       round(vp / total_vp, 4),
            # Cards
            "dev_played":     dev_played[p],
            "knights_played": knights[p],
            "dev_in_hand":    dev_in_hand[p],
            "hand_size":      hand_size[p],
            # Roads / robber
            "road_count":     road_count[p],
            "times_robbed":   robbed[p],
            "robber_on_my_tile": on_rob,
            # Achievements
            "la_flag":        la_flag[p],
            "lr_flag":        lr_flag[p],
            # Ports
            "port_generic":   int(1 in pp),
            "port_brick":     int(2 in pp),
            "port_grain":     int(3 in pp),
            "port_wool":      int(4 in pp),
            "port_lumber":    int(5 in pp),
            "port_ore":       int(6 in pp),
            # Meta
            "total_turns":    total_turns,
            "rank":           final_rank.get(p, 4),
            "is_winner":      int(winner_color == p),
        })
    return rows


out_path   = Path("features_full.csv")
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

        # ── Build per-game static lookups from initialState ────────────────────
        init_map = hist.get("initialState", {}).get("mapState", {})

        # Hex: (x, y) → (diceNumber, resourceType)
        hex_lookup = {}
        for hd in init_map.get("tileHexStates", {}).values():
            x, y = hd.get("x"), hd.get("y")
            if x is not None and y is not None:
                hex_lookup[(x, y)] = (hd.get("diceNumber"), hd.get("type", 0))

        # Corner: corner_id → (x, y, z)
        corner_xy = {}
        for cid, cd in init_map.get("tileCornerStates", {}).items():
            corner_xy[cid] = (cd.get("x"), cd.get("y"), cd.get("z"))

        # Port: (x, y, z) → port_type
        port_at_coord = {}
        for pd in init_map.get("portEdgeStates", {}).values():
            port_at_coord[(pd.get("x"), pd.get("y"), pd.get("z"))] = pd.get("type")

        # ── Per-player mutable accumulators ───────────────────────────────────
        all_player_corners = defaultdict(dict)   # color → {cid: buildingType}
        income             = defaultdict(int)
        income_by_rt       = defaultdict(lambda: defaultdict(int))
        dev_played         = defaultdict(int)
        knights            = defaultdict(int)
        robbed             = defaultdict(int)
        dev_in_hand        = defaultdict(int)
        hand_size          = defaultdict(int)
        road_count         = defaultdict(int)
        edge_seen          = set()               # to avoid double-counting road edges
        la_flag            = defaultdict(int)
        lr_flag            = defaultdict(int)
        player_ports       = defaultdict(set)    # color → set of port types
        robber_pos         = None                # current robber (x, y)

        cur_turns          = 0
        last_snapshot_turn = -1

        for ev in hist["events"]:
            sc = ev.get("stateChange", {})

            # ── Turn counter & snapshot emission ──────────────────────────────
            cs_state = sc.get("currentState", {})
            if "completedTurns" in cs_state:
                cur_turns = cs_state["completedTurns"]
                next_snap = last_snapshot_turn + SNAPSHOT_INTERVAL
                while next_snap <= cur_turns:
                    if next_snap > 0:
                        for row in make_rows(
                            next_snap, gf.stem, play_order, players,
                            all_player_corners, income, income_by_rt,
                            dev_played, knights, robbed, dev_in_hand, hand_size,
                            road_count, la_flag, lr_flag, player_ports,
                            corner_xy, hex_lookup, robber_pos,
                            total_turns, final_rank, winner_color
                        ):
                            writer.writerow(row)
                            total_rows += 1
                    last_snapshot_turn = next_snap
                    next_snap += SNAPSHOT_INTERVAL

            # ── Map state ─────────────────────────────────────────────────────
            ms = sc.get("mapState", {})

            for cid, cdata in ms.get("tileCornerStates", {}).items():
                if "owner" not in cdata:
                    continue
                owner = cdata["owner"]
                bt    = cdata.get("buildingType", 0)
                if bt not in (1, 2):
                    continue
                all_player_corners[owner][cid] = bt
                # Port detection
                xyz = corner_xy.get(cid)
                if xyz:
                    pt = port_at_coord.get(xyz)
                    if pt:
                        player_ports[owner].add(pt)

            for eid, edata in ms.get("tileEdgeStates", {}).items():
                if "owner" in edata and eid not in edge_seen:
                    edge_seen.add(eid)
                    road_count[edata["owner"]] += 1

            # ── Robber position ────────────────────────────────────────────────
            rs = sc.get("mechanicRobberState", {})
            rx, ry = rs.get("x"), rs.get("y")
            if rx is not None and ry is not None:
                robber_pos = (rx, ry)

            # ── Player states (LA/LR flags, hand size) ─────────────────────────
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
                rc = pstate.get("resourceCards", {})
                if "cards" in rc:
                    hand_size[color] = len(rc["cards"])

            # ── Dev card hand count ────────────────────────────────────────────
            mdc = sc.get("mechanicDevelopmentCardsState", {})
            for cs_str, pdc in mdc.get("players", {}).items():
                try:
                    color = int(cs_str)
                except ValueError:
                    continue
                cards = pdc.get("developmentCards", {}).get("cards")
                if cards is not None:
                    dev_in_hand[color] = len(cards)

            # ── Game log events ────────────────────────────────────────────────
            for log in sc.get("gameLogState", {}).values():
                t  = log.get("text", {})
                tt = t.get("type")
                pc = t.get("playerColor")

                if tt == 11 and pc:                       # resource income from roll
                    income[pc] += 1
                    rt = t.get("tileInfo", {}).get("resourceType")
                    if rt:
                        income_by_rt[pc][rt] += 1
                elif tt == 20 and pc:                     # dev card played
                    dev_played[pc] += 1
                    if t.get("cardEnum") in (10, 11):     # knight
                        knights[pc] += 1
                elif tt == 16:                            # steal
                    victim = t.get("playerColorVictim")
                    if victim:
                        robbed[victim] += 1

        # Final snapshot
        if cur_turns != last_snapshot_turn and cur_turns > 0:
            for row in make_rows(
                cur_turns, gf.stem, play_order, players,
                all_player_corners, income, income_by_rt,
                dev_played, knights, robbed, dev_in_hand, hand_size,
                road_count, la_flag, lr_flag, player_ports,
                corner_xy, hex_lookup, robber_pos,
                total_turns, final_rank, winner_color
            ):
                writer.writerow(row)
                total_rows += 1

print(f"Done. {total_rows:,} rows written to {out_path}  ({skipped} games skipped)")
