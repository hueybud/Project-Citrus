#!/usr/bin/env python3
"""
Compare two CITF files frame-by-frame to diagnose deterministic replay divergence.

Usage:
    python compare_citf_citf.py <reference.citframes> <replay.citframes>

The first file is treated as the ground truth (reference). The script:
  - Compares all game state fields per frame (positions, scores, ball state,
    character states, controllers, items)
  - Reports the first frame of divergence with a full field-by-field diff
  - Summarises which fields diverge across all mismatched frames
  - Indicates whether the divergence cascades or is transient

NOTE: Ball owner pointers are resolved to slot indices before comparison because
heap addresses legitimately differ between Dolphin sessions (fresh boot vs rematch).
Controllers are compared directly to verify input injection is working.
"""

import struct
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Enums (mirrors analyze_citf.py)
# ---------------------------------------------------------------------------

CAPTAINS = {0: "Daisy", 1: "DK", 2: "Luigi", 3: "Mario", 4: "Peach",
            5: "Waluigi", 6: "Wario", 7: "Yoshi", 8: "SuperTeam"}

SIDEKICKS = {0: "Toad", 1: "Koopa", 2: "HammerBro", 3: "Birdo", 8: "SuperTeam"}

STADIUMS = {0: "The Pipeline", 1: "The Palace", 2: "Konga Coliseum",
            3: "The Underground", 4: "Crater Field", 5: "Bowser Stadium",
            6: "Battle Dome"}

GAME_PHASE_NAMES = {
    0: "Pre-match", 1: "Kickoff", 2: "Goal celebration",
    3: "Transition", 4: "Active play", 5: "Active play (2)",
}

STRIKER_ACTION_STATES = {
    0x00: "Deke", 0x01: "Electrocuted (wall)", 0x02: "Defensive hit",
    0x03: "Hit by opponent", 0x04: "Idle turn", 0x05: "Air shot",
    0x06: "Receiving pass (toad)", 0x08: "Receiving perfect pass",
    0x0A: "Initiating pass", 0x0C: "Receiving pass", 0x0D: "Idle",
    0x0E: "Ball carrier", 0x0F: "Offensive sprint", 0x11: "Ground shot",
    0x13: "Defensive slide", 0x14: "Bumped/body-checked",
    0x15: "Exploding (item)", 0x16: "Hit by item", 0x17: "Slipped on banana",
    0x19: "Hit by big shell", 0x1A: "Captain slide (no ball)",
    0x1B: "Pre-perfect-pass", 0xFF: "State clearing",
}

GOALIE_ACTION_STATES = {
    0x00: "Default/idle", 0x01: "Walking with ball", 0x02: "Hands out",
    0x03: "Preparing for charged shot", 0x04: "Jumping/diving to ball",
    0x06: "Laying down (after contact)", 0x09: "Stunned by ball",
    0x0C: "Throwing ball", 0x11: "Jogging to ball",
    0x13: "Holding ball", 0x15: "Walking to pick up ball",
    0x17: "Laying down (goal allowed)",
}

PAD_BUTTONS = {
    0x0001: "LEFT", 0x0002: "RIGHT", 0x0004: "DOWN", 0x0008: "UP",
    0x0010: "Z", 0x0020: "L", 0x0040: "R", 0x0100: "A", 0x0200: "B",
    0x0400: "X", 0x0800: "Y", 0x1000: "START",
}

BASE_HEADER_SIZE = 48    # v7-v10
V11_HEADER_SIZE  = 273  # v11+ (48-byte base + 225-byte match metadata)
CHAR_SIZE        = 28   # sizeof(FrameCharacter)
CTRL_SIZE        = 10   # sizeof(FrameControllerInput)
INVENTORY_SIZE   = 8    # sizeof(FramePowerupInventorySlot)
ITEM_SIZE        = 48   # sizeof(FrameItem)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def char_slot_name(idx):
    if idx < 4:
        return f"L_striker{idx}"
    elif idx == 4:
        return "L_goalie"
    elif idx < 9:
        return f"R_striker{idx - 5}"
    else:
        return "R_goalie"


def action_state_name(slot_idx, state):
    if slot_idx in (4, 9):
        return GOALIE_ACTION_STATES.get(state, f"Unknown(0x{state:02X})")
    return STRIKER_ACTION_STATES.get(state, f"Unknown(0x{state:02X})")


def buttons_str(btn):
    parts = [name for mask, name in sorted(PAD_BUTTONS.items()) if btn & mask]
    return "+".join(parts) if parts else "none"


def resolve_owner(char_ptrs, owner_ptr):
    """Resolve a raw pointer to a slot index (None if null, -1 if unresolved)."""
    if owner_ptr == 0:
        return None
    for i, ptr in enumerate(char_ptrs):
        if ptr == owner_ptr:
            return i
    return -1  # non-null pointer not found in char_ptrs


def owner_label(slot):
    """Human-readable label for a resolved owner slot."""
    if slot is None:
        return "NONE"
    if slot == -1:
        return "UNRESOLVED"
    return char_slot_name(slot)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_file_header(data):
    """Parse CITF header (v7–v11). Returns a dict."""
    magic = data[0:4]
    if magic != b'CITF':
        raise ValueError(f"Invalid CITF magic: {magic!r}")

    version, frame_count, fixed_frame_size = struct.unpack_from('<III', data, 4)

    left_captain   = data[16]
    right_captain  = data[17]
    left_sidekick  = data[18]
    right_sidekick = data[19]
    stadium_id     = data[20]
    # 3 padding bytes at 21-23

    goal_line_x, sideline_y, penalty_box_x, net_half_width, net_height, net_depth = \
        struct.unpack_from('<6f', data, 24)

    return {
        'version':          version,
        'frame_count':      frame_count,
        'fixed_frame_size': fixed_frame_size,
        'left_captain':     left_captain,
        'right_captain':    right_captain,
        'left_sidekick':    left_sidekick,
        'right_sidekick':   right_sidekick,
        'stadium_id':       stadium_id,
        'goal_line_x':      goal_line_x,
        'sideline_y':       sideline_y,
        'penalty_box_x':    penalty_box_x,
        'net_half_width':   net_half_width,
        'net_height':       net_height,
        'net_depth':        net_depth,
        # v11+ adds 225-byte match metadata block; header_size reflects actual frame data offset
        'header_size':      V11_HEADER_SIZE if version >= 11 else BASE_HEADER_SIZE,
    }


def parse_frame(data, offset, fixed_size):
    """Parse one GameStateFrame. Returns (frame_dict, bytes_consumed)."""
    o = offset

    game_time, movie_frame = struct.unpack_from('<fI', data, o); o += 8
    left_score, right_score, is_paused, game_phase = struct.unpack_from('<BBBB', data, o); o += 4

    # Ball state (size depends on version)
    bpx, bpy, bpz, bvx, bvy, bvz = struct.unpack_from('<ffffff', data, o); o += 24
    ball_owner = struct.unpack_from('<I', data, o)[0]; o += 4

    potential_scorer = 0
    if fixed_size >= 472:  # v9+
        potential_scorer = struct.unpack_from('<I', data, o)[0]; o += 4

    ball_pass_target = 0
    if fixed_size >= 476:  # v10+
        ball_pass_target = struct.unpack_from('<I', data, o)[0]; o += 4

    is_perfect_pass = data[o]; o += 4   # 1 byte + 3 padding
    ball_charge = struct.unpack_from('<f', data, o)[0]; o += 4

    # Characters (10 x 28 bytes)
    chars = []
    for _ in range(10):
        px, py, pz = struct.unpack_from('<fff', data, o)
        action_state  = struct.unpack_from('<I', data, o + 12)[0]
        heading       = struct.unpack_from('<H', data, o + 16)[0]
        effect_type   = data[o + 18]
        speed_item_type  = data[o + 19]
        speed_item_count = data[o + 20]
        is_user_ctrl  = data[o + 21]
        speed_timer   = struct.unpack_from('<f', data, o + 24)[0]
        chars.append({
            'pos_x': px, 'pos_y': py, 'pos_z': pz,
            'action_state': action_state,
            'heading': heading,
            'effect_type': effect_type,
            'speed_item_type': speed_item_type,
            'speed_item_count': speed_item_count,
            'is_user_controlled': is_user_ctrl,
            'speed_item_timer': speed_timer,
        })
        o += CHAR_SIZE

    # Character pointers (10 x 4 bytes)
    char_ptrs = []
    for _ in range(10):
        char_ptrs.append(struct.unpack_from('<I', data, o)[0])
        o += 4

    # Controllers (4 x 10 bytes)
    ctrls = []
    for _ in range(4):
        buttons = struct.unpack_from('<H', data, o)[0]
        ctrls.append({
            'buttons':      buttons,
            'stick_x':      data[o + 2],
            'stick_y':      data[o + 3],
            'substick_x':   data[o + 4],
            'substick_y':   data[o + 5],
            'trigger_left': data[o + 6],
            'trigger_right':data[o + 7],
            'is_connected': data[o + 8],
        })
        o += CTRL_SIZE

    # Powerup inventory (4 slots x 8 bytes)
    inventory = []
    for _ in range(4):
        inv_type     = struct.unpack_from('<i', data, o)[0]
        charge_count = data[o + 4]
        is_new       = data[o + 5]
        inventory.append({'type': inv_type, 'charge_count': charge_count, 'is_new': is_new})
        o += INVENTORY_SIZE

    # Team stats (v8+: 24 bytes)
    left_stats = right_stats = None
    if fixed_size >= 468:
        ls = struct.unpack_from('<5H', data, o); o += 10
        rs = struct.unpack_from('<5H', data, o); o += 10
        o += 4  # statsPadding
        keys = ('shots', 'hits', 'steals', 'super_strikes', 'perfect_passes')
        left_stats  = dict(zip(keys, ls))
        right_stats = dict(zip(keys, rs))

    # Item count + padding (4 bytes)
    item_count = data[o]; o += 4

    # Items (item_count x 48 bytes)
    items = []
    for _ in range(item_count):
        vals = struct.unpack_from('<ffffffBBBBIIIfHH', data, o)
        items.append({
            'pos_x': vals[0], 'pos_y': vals[1], 'pos_z': vals[2],
            'vel_x': vals[3], 'vel_y': vals[4], 'vel_z': vals[5],
            'powerup_type': vals[6], 'strength_level': vals[7],
            'slot_index': vals[8],
        })
        o += ITEM_SIZE

    return {
        'game_time':           game_time,
        'movie_frame':         movie_frame,
        'left_score':          left_score,
        'right_score':         right_score,
        'is_paused':           is_paused,
        'game_phase':          game_phase,
        'ball_pos_x':          bpx,
        'ball_pos_y':          bpy,
        'ball_pos_z':          bpz,
        'ball_vel_x':          bvx,
        'ball_vel_y':          bvy,
        'ball_vel_z':          bvz,
        'ball_owner_ptr':      ball_owner,
        'potential_scorer_ptr':potential_scorer,
        'ball_pass_target_ptr':ball_pass_target,
        'is_perfect_pass':     is_perfect_pass,
        'ball_charge':         ball_charge,
        'characters':          chars,
        'char_ptrs':           char_ptrs,
        'controllers':         ctrls,
        'inventory':           inventory,
        'left_stats':          left_stats,
        'right_stats':         right_stats,
        'item_count':          item_count,
        'items':               items,
    }, o - offset


def load_citf(path):
    """Load and parse a CITF file. Returns (header_dict, list[frame_dict])."""
    with open(path, 'rb') as f:
        data = f.read()

    header = parse_file_header(data)
    fixed_size = header['fixed_frame_size']

    frames = []
    offset = header['header_size']
    for _ in range(header['frame_count']):
        frame, consumed = parse_frame(data, offset, fixed_size)
        frames.append(frame)
        offset += consumed

    trailing = len(data) - offset
    if trailing != 0:
        print(f"  WARNING: {trailing} trailing bytes in {Path(path).name}")

    return header, frames


# ---------------------------------------------------------------------------
# Frame comparison
# ---------------------------------------------------------------------------

POS_TOLERANCE    = 1e-4
VEL_TOLERANCE    = 1e-4
CHARGE_TOLERANCE = 1e-4


def compare_frame_pair(ref, rep):
    """
    Compare two frames field by field.
    Returns list of (field_name, ref_val, rep_val) for every mismatch.
    Ball owner is resolved to a slot index before comparison.
    """
    diffs = []

    def chk(name, a, b, tol=None):
        if tol is not None:
            if abs(float(a) - float(b)) > tol:
                diffs.append((name, a, b))
        else:
            if a != b:
                diffs.append((name, a, b))

    # --- Metadata / score ---
    chk('game_phase',   ref['game_phase'],  rep['game_phase'])
    chk('left_score',   ref['left_score'],  rep['left_score'])
    chk('right_score',  ref['right_score'], rep['right_score'])
    chk('is_paused',    ref['is_paused'],   rep['is_paused'])
    chk('game_time',    ref['game_time'],   rep['game_time'],  tol=POS_TOLERANCE)
    chk('movie_frame',  ref['movie_frame'], rep['movie_frame'])

    # --- Ball ---
    chk('ball_pos_x', ref['ball_pos_x'], rep['ball_pos_x'], tol=POS_TOLERANCE)
    chk('ball_pos_y', ref['ball_pos_y'], rep['ball_pos_y'], tol=POS_TOLERANCE)
    chk('ball_pos_z', ref['ball_pos_z'], rep['ball_pos_z'], tol=POS_TOLERANCE)
    chk('ball_vel_x', ref['ball_vel_x'], rep['ball_vel_x'], tol=VEL_TOLERANCE)
    chk('ball_vel_y', ref['ball_vel_y'], rep['ball_vel_y'], tol=VEL_TOLERANCE)
    chk('ball_vel_z', ref['ball_vel_z'], rep['ball_vel_z'], tol=VEL_TOLERANCE)
    chk('is_perfect_pass', ref['is_perfect_pass'], rep['is_perfect_pass'])
    chk('ball_charge',     ref['ball_charge'],     rep['ball_charge'], tol=CHARGE_TOLERANCE)

    # Ball owner: compare resolved slot index to avoid heap-address differences
    ref_slot = resolve_owner(ref['char_ptrs'], ref['ball_owner_ptr'])
    rep_slot = resolve_owner(rep['char_ptrs'], rep['ball_owner_ptr'])
    if ref_slot != rep_slot:
        # If both are unresolved (-1) but different raw pointers, also note raw diff
        ref_lbl = owner_label(ref_slot) if ref_slot != -1 else f"UNRESOLVED(0x{ref['ball_owner_ptr']:08X})"
        rep_lbl = owner_label(rep_slot) if rep_slot != -1 else f"UNRESOLVED(0x{rep['ball_owner_ptr']:08X})"
        diffs.append(('ball_owner_slot', ref_lbl, rep_lbl))
    elif ref_slot == -1 and rep_slot == -1:
        # Both unresolved — compare raw pointers as fallback
        if ref['ball_owner_ptr'] != rep['ball_owner_ptr']:
            diffs.append(('ball_owner_raw_ptr',
                          f"0x{ref['ball_owner_ptr']:08X}",
                          f"0x{rep['ball_owner_ptr']:08X}"))

    # --- Characters (10 slots) ---
    for i in range(10):
        rc  = ref['characters'][i]
        rpc = rep['characters'][i]
        pfx = char_slot_name(i)
        chk(f'{pfx}.pos_x',          rc['pos_x'],          rpc['pos_x'],          tol=POS_TOLERANCE)
        chk(f'{pfx}.pos_y',          rc['pos_y'],          rpc['pos_y'],          tol=POS_TOLERANCE)
        chk(f'{pfx}.pos_z',          rc['pos_z'],          rpc['pos_z'],          tol=POS_TOLERANCE)
        chk(f'{pfx}.action_state',   rc['action_state'],   rpc['action_state'])
        chk(f'{pfx}.heading',        rc['heading'],        rpc['heading'])
        chk(f'{pfx}.effect_type',    rc['effect_type'],    rpc['effect_type'])
        chk(f'{pfx}.speed_item_type',rc['speed_item_type'],rpc['speed_item_type'])

    # --- Controllers (4 ports) ---
    for p in range(4):
        rc  = ref['controllers'][p]
        rpc = rep['controllers'][p]
        pfx = f'ctrl[{p}]'
        chk(f'{pfx}.buttons',      rc['buttons'],      rpc['buttons'])
        chk(f'{pfx}.stick_x',      rc['stick_x'],      rpc['stick_x'])
        chk(f'{pfx}.stick_y',      rc['stick_y'],      rpc['stick_y'])
        chk(f'{pfx}.substick_x',   rc['substick_x'],   rpc['substick_x'])
        chk(f'{pfx}.substick_y',   rc['substick_y'],   rpc['substick_y'])
        chk(f'{pfx}.trigger_left', rc['trigger_left'], rpc['trigger_left'])
        chk(f'{pfx}.trigger_right',rc['trigger_right'],rpc['trigger_right'])

    # Item count (items are RNG-driven, count divergence is meaningful)
    chk('item_count', ref['item_count'], rep['item_count'])

    return diffs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def fmt_diff_line(name, ref_val, rep_val):
    if isinstance(ref_val, float):
        return (f"  {name:<42}  ref={ref_val:12.6f}  rep={rep_val:12.6f}"
                f"  delta={rep_val - ref_val:+.6f}")
    elif isinstance(ref_val, int):
        return (f"  {name:<42}  ref={ref_val:>10}  rep={rep_val:>10}"
                f"  delta={rep_val - ref_val:+d}")
    else:
        return f"  {name:<42}  ref={ref_val!s}  rep={rep_val!s}"


def frame_oneliner(label, frame):
    slot = resolve_owner(frame['char_ptrs'], frame['ball_owner_ptr'])
    phase = GAME_PHASE_NAMES.get(frame['game_phase'], f"?({frame['game_phase']})")
    return (f"  {label}:  t={frame['game_time']:.2f}s"
            f"  score={frame['left_score']}-{frame['right_score']}"
            f"  phase={phase}"
            f"  ball=({frame['ball_pos_x']:.2f},{frame['ball_pos_y']:.2f},{frame['ball_pos_z']:.2f})"
            f"  owner={owner_label(slot)}"
            f"  charge={frame['ball_charge']:.1f}")


def print_character_states(frame, label=""):
    if label:
        print(f"\n  Character states{label}:")
    for i in range(10):
        c = frame['characters'][i]
        state_name = action_state_name(i, c['action_state'])
        ctrl_flag  = " [HUMAN]" if c['is_user_controlled'] else ""
        print(f"    {char_slot_name(i):<12}  0x{c['action_state']:02X} {state_name:<28}"
              f"  pos=({c['pos_x']:7.2f},{c['pos_y']:7.2f},{c['pos_z']:5.2f})"
              f"  eff={c['effect_type']}{ctrl_flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <reference.citframes> <replay.citframes>")
        print()
        print("Compares two CITF files frame-by-frame to diagnose deterministic replay divergence.")
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    rep_path = Path(sys.argv[2])

    for p in (ref_path, rep_path):
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    SEP = "=" * 90
    sep = "-" * 90

    print(SEP)
    print("CITF vs CITF COMPARISON")
    print(SEP)

    print(f"\nLoading reference : {ref_path.name}")
    ref_header, ref_frames = load_citf(ref_path)
    print(f"  v{ref_header['version']}  {ref_header['frame_count']} frames"
          f"  fixed_size={ref_header['fixed_frame_size']}  header={ref_header['header_size']}B"
          f"  file={ref_path.stat().st_size:,} bytes")

    print(f"\nLoading replay    : {rep_path.name}")
    rep_header, rep_frames = load_citf(rep_path)
    print(f"  v{rep_header['version']}  {rep_header['frame_count']} frames"
          f"  fixed_size={rep_header['fixed_frame_size']}  header={rep_header['header_size']}B"
          f"  file={rep_path.stat().st_size:,} bytes")

    # -----------------------------------------------------------------------
    # Header comparison
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("HEADER COMPARISON")
    print(sep)

    def hdr_row(name, a, b):
        ok = "OK  " if a == b else "DIFF"
        print(f"  [{ok}] {name:<22} ref={str(a):<28} rep={str(b)}")

    hdr_row("version",         ref_header['version'],         rep_header['version'])
    hdr_row("frame_count",     ref_header['frame_count'],     rep_header['frame_count'])
    hdr_row("fixed_frame_size",ref_header['fixed_frame_size'],rep_header['fixed_frame_size'])

    ref_left  = f"{CAPTAINS.get(ref_header['left_captain'],'?')} + {SIDEKICKS.get(ref_header['left_sidekick'],'?')}"
    rep_left  = f"{CAPTAINS.get(rep_header['left_captain'],'?')} + {SIDEKICKS.get(rep_header['left_sidekick'],'?')}"
    ref_right = f"{CAPTAINS.get(ref_header['right_captain'],'?')} + {SIDEKICKS.get(ref_header['right_sidekick'],'?')}"
    rep_right = f"{CAPTAINS.get(rep_header['right_captain'],'?')} + {SIDEKICKS.get(rep_header['right_sidekick'],'?')}"
    hdr_row("left team",  ref_left,  rep_left)
    hdr_row("right team", ref_right, rep_right)
    hdr_row("stadium",
            STADIUMS.get(ref_header['stadium_id'], f"?({ref_header['stadium_id']})"),
            STADIUMS.get(rep_header['stadium_id'], f"?({rep_header['stadium_id']})"))

    teams_ok = (ref_header['left_captain']  == rep_header['left_captain']  and
                ref_header['right_captain'] == rep_header['right_captain'] and
                ref_header['left_sidekick'] == rep_header['left_sidekick'] and
                ref_header['right_sidekick']== rep_header['right_sidekick'] and
                ref_header['stadium_id']    == rep_header['stadium_id'])
    if not teams_ok:
        print("\n  WARNING: Headers describe DIFFERENT matches — results may not be meaningful.")

    # -----------------------------------------------------------------------
    # Frame-by-frame comparison
    # -----------------------------------------------------------------------
    n = min(len(ref_frames), len(rep_frames))
    if len(ref_frames) != len(rep_frames):
        print(f"\n  NOTE: Frame count differs ({len(ref_frames)} vs {len(rep_frames)})"
              f" — comparing first {n} frames.")

    print(f"\n{sep}")
    print(f"FRAME-BY-FRAME COMPARISON  ({n} frames)")
    print(sep)

    all_diffs      = []   # per-frame list of diffs (empty list = identical)
    first_div      = None
    total_div      = 0
    field_counts   = {}   # field_name -> count of frames where it diverges
    consec_div     = 0
    max_consec_div = 0
    consec_ok      = 0
    recovered_at   = None

    for i in range(n):
        diffs = compare_frame_pair(ref_frames[i], rep_frames[i])
        all_diffs.append(diffs)

        if diffs:
            total_div += 1
            if first_div is None:
                first_div = i
            consec_div += 1
            consec_ok   = 0
            max_consec_div = max(max_consec_div, consec_div)
            for name, _, _ in diffs:
                field_counts[name] = field_counts.get(name, 0) + 1
        else:
            consec_div = 0
            consec_ok += 1
            if first_div is not None and recovered_at is None and consec_ok >= 5:
                recovered_at = i  # sustained recovery

    # -----------------------------------------------------------------------
    # Overall result
    # -----------------------------------------------------------------------
    if first_div is None:
        print("\n")
        print(SEP)
        print("  RESULT: FILES ARE IDENTICAL (all compared fields match across all frames)")
        print(SEP)
        return 0

    identical = n - total_div
    print(f"\n  Frames compared   : {n:>7}")
    print(f"  Identical frames  : {identical:>7}  ({100 * identical / n:.1f}%)")
    print(f"  Diverged frames   : {total_div:>7}  ({100 * total_div / n:.1f}%)")
    print(f"  First divergence  : frame {first_div}"
          f"  (game_time={ref_frames[first_div]['game_time']:.2f}s"
          f"  movie_frame={ref_frames[first_div]['movie_frame']})")
    print(f"  Max consec. diff  : {max_consec_div} frames")
    if recovered_at is not None:
        print(f"  Recovery at frame : {recovered_at}  (transient divergence — resync'd)")
    else:
        print(f"  Recovery          : NONE (cascading — divergence never corrects itself)")

    # -----------------------------------------------------------------------
    # First divergence: full detail
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print(f"FIRST DIVERGENCE — Frame {first_div}")
    print(sep)

    ref_f0 = ref_frames[first_div]
    rep_f0 = rep_frames[first_div]

    print()
    print(frame_oneliner("REF", ref_f0))
    print(frame_oneliner("REP", rep_f0))
    print(f"\n  Divergent fields ({len(all_diffs[first_div])}):")
    for name, rv, rpv in all_diffs[first_div]:
        print(fmt_diff_line(name, rv, rpv))

    # Context: frames leading up to the divergence
    CONTEXT = 6
    look_start = max(0, first_div - CONTEXT)
    if look_start < first_div:
        print(f"\n  Context ({first_div - look_start} frames before first divergence) [ref side]:")
        print(f"  {'F':>6}  {'t':>7}  {'phase':<20}  {'score':>6}  {'owner':<14}  charge")
        for i in range(look_start, first_div):
            f = ref_frames[i]
            slot = resolve_owner(f['char_ptrs'], f['ball_owner_ptr'])
            phase = GAME_PHASE_NAMES.get(f['game_phase'], f"?({f['game_phase']})")
            print(f"  {i:>6}  {f['game_time']:6.2f}s  {phase:<20}  "
                  f"{f['left_score']}-{f['right_score']:>2}  {owner_label(slot):<14}  {f['ball_charge']:.1f}")

    # Show character states at first divergence (both sides)
    print_character_states(ref_f0, label=f" [REF] at frame {first_div}")
    print_character_states(rep_f0, label=f" [REP] at frame {first_div}")

    # -----------------------------------------------------------------------
    # Next 12 frames — status at a glance
    # -----------------------------------------------------------------------
    SHOW = 12
    print(f"\n  Next {SHOW} frames after first divergence:")
    print(f"  {'F':>6}  {'status':<12}  {'t':>7}  {'phase':<20}  score  owner")
    for i in range(first_div + 1, min(first_div + 1 + SHOW, n)):
        diffs = all_diffs[i]
        status = f"DIFF({len(diffs)})" if diffs else "OK"
        f = ref_frames[i]
        slot = resolve_owner(f['char_ptrs'], f['ball_owner_ptr'])
        phase = GAME_PHASE_NAMES.get(f['game_phase'], f"?")
        print(f"  {i:>6}  [{status:<9}]  {f['game_time']:6.2f}s  {phase:<20}  "
              f"{f['left_score']}-{f['right_score']}  {owner_label(slot)}")

    # -----------------------------------------------------------------------
    # Field divergence summary (sorted by frequency)
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("FIELD DIVERGENCE SUMMARY")
    print(sep)
    print(f"\n  {'Field':<42}  {'Frames':>8}  {'%':>6}")
    print(f"  {'─'*42}  {'─'*8}  {'─'*6}")
    for name, cnt in sorted(field_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / n
        bar = "#" * min(20, int(pct / 5))
        print(f"  {name:<42}  {cnt:>8}  {pct:5.1f}%  {bar}")

    # --- Character divergence rollup ---
    char_div = {i: 0 for i in range(10)}
    for diffs in all_diffs:
        seen_slots = set()
        for name, _, _ in diffs:
            dot = name.find('.')
            if dot == -1:
                continue
            slot_str = name[:dot]
            for si in range(10):
                if char_slot_name(si) == slot_str:
                    seen_slots.add(si)
                    break
        for si in seen_slots:
            char_div[si] += 1

    has_char_div = any(v > 0 for v in char_div.values())
    if has_char_div:
        print(f"\n  Character divergence rollup (frames where any field of that slot differs):")
        for si in range(10):
            if char_div[si] > 0:
                pct = 100 * char_div[si] / n
                print(f"    {char_slot_name(si):<12}  {char_div[si]:>6} frames  ({pct:.1f}%)")

    # --- Controller check ---
    ctrl_div_frames = sum(1 for diffs in all_diffs
                          if any(name.startswith('ctrl') for name, _, _ in diffs))
    print()
    if ctrl_div_frames == 0:
        print(f"  Controllers : MATCH — inputs are identical across all frames")
    else:
        print(f"  Controllers : DIVERGE in {ctrl_div_frames} frames"
              f"  — input injection may not be working correctly")
        # Show first controller divergence detail
        for i, diffs in enumerate(all_diffs):
            ctrl_diffs = [(n, rv, rpv) for n, rv, rpv in diffs if n.startswith('ctrl')]
            if ctrl_diffs:
                print(f"    First ctrl diff at frame {i}:")
                for name, rv, rpv in ctrl_diffs[:6]:
                    ref_btn_str = f"  ({buttons_str(rv)})" if 'buttons' in name and isinstance(rv, int) else ""
                    rep_btn_str = f"  ({buttons_str(rpv)})" if 'buttons' in name and isinstance(rpv, int) else ""
                    print(f"      {fmt_diff_line(name, rv, rpv)}{ref_btn_str}")
                break

    # --- Score divergence ---
    score_div_frames = [(i, ref_frames[i], rep_frames[i])
                        for i, diffs in enumerate(all_diffs)
                        if any(n in ('left_score', 'right_score') for n, _, _ in diffs)]
    if score_div_frames:
        print(f"\n  Score diverges at {len(score_div_frames)} frame(s) (first 5):")
        for i, rf, rpf in score_div_frames[:5]:
            print(f"    F{i:5d}  ref={rf['left_score']}-{rf['right_score']}"
                  f"  rep={rpf['left_score']}-{rpf['right_score']}")
    else:
        print(f"  Score     : MATCH across all frames")

    # --- Item count divergence ---
    item_div_frames = [(i, ref_frames[i]['item_count'], rep_frames[i]['item_count'])
                       for i in range(n)
                       if ref_frames[i]['item_count'] != rep_frames[i]['item_count']]
    if item_div_frames:
        print(f"\n  Item count diverges at {len(item_div_frames)} frame(s) (first 5 shown):")
        print(f"  This indicates RNG divergence — powerup spawns differ between runs.")
        for i, rc, rpc in item_div_frames[:5]:
            print(f"    F{i:5d}  ref={rc} items  rep={rpc} items"
                  f"  (t={ref_frames[i]['game_time']:.2f}s)")
    else:
        print(f"  Item count: MATCH across all frames")

    # -----------------------------------------------------------------------
    # Diagnosis hint
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("DIAGNOSIS HINTS")
    print(sep)

    has_ctrl_mismatch = ctrl_div_frames > 0
    has_item_mismatch = len(item_div_frames) > 0
    has_score_mismatch = len(score_div_frames) > 0

    if has_ctrl_mismatch:
        print("\n  [!] CONTROLLER INPUTS DIVERGE: Injection is broken — the replay game")
        print("      is not receiving the same inputs as the reference. Check PlayController")
        print("      CITF path in Movie.cpp and the s_use_citf_inputs flag.")

    if not has_ctrl_mismatch and has_item_mismatch and first_div is not None:
        print(f"\n  [2] INPUTS MATCH but items diverge at frame {item_div_frames[0][0]}.")
        print("      Likely cause: RNG state differs despite seed injection. Possibilities:")
        print("        a) Seed injected too late (after an RNG draw already occurred).")
        print("        b) Rematch path fires extra RNG draws before addressMatchStart.")
        print("        c) Seed captured at wrong point in reference (should be at SetSlideOut).")

    if not has_ctrl_mismatch and not has_item_mismatch and first_div is not None:
        print(f"\n  [3] INPUTS AND ITEMS MATCH but positions diverge at frame {first_div}.")
        print("      The deterministic inputs produce different physics. Investigate:")
        print("        a) Is the diverging character's position off? Check action states above.")
        print("        b) Ball velocity differs? Could be initial ball placement (rematch path).")
        print("        c) Check if first divergence is at a kickoff frame (phase 1).")

    if not has_rng_mismatch and not has_ctrl_mismatch and not has_item_mismatch and not has_score_mismatch:
        print("\n  [4] No obvious root cause. All diagnostics pass except position/state.")
        print("      Consider comparing ball velocity at frame 0 and character initial positions.")

    print(f"\n{SEP}")
    print("COMPARISON COMPLETE")
    print(SEP)
    return 1  # non-zero = mismatch found


if __name__ == "__main__":
    sys.exit(main())
