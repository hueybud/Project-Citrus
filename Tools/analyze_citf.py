#!/usr/bin/env python3
"""
Analyze a CITF v10 (Citrus Input/Telemetry Frame) file.
Produces scoring narrative with shot classification, action state analysis,
game phase breakdown, and match statistics.
"""

import struct
import sys
import math
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from dataclasses import dataclass, field
from typing import List, Optional

# -- Enums --------------------------------------------------------------------

CAPTAINS = {0: "Daisy", 1: "DK", 2: "Luigi", 3: "Mario", 4: "Peach",
            5: "Waluigi", 6: "Wario", 7: "Yoshi", 8: "SuperTeam"}

SIDEKICKS = {0: "Toad", 1: "Koopa", 2: "HammerBro", 3: "Birdo", 8: "SuperTeam"}

STADIUMS = {0: "The Pipeline", 1: "The Palace", 2: "Konga Coliseum",
            3: "The Underground", 4: "Crater Field", 5: "Bowser Stadium",
            6: "Battle Dome"}

POWERUP_TYPES = {-1: "Empty", 0: "Green Shell", 1: "Red Shell", 2: "Shell",
                 3: "Blue Shell", 4: "Banana", 5: "Bob-omb"}

EFFECT_NAMES = {0: "none", 1: "frozen", 2: "on_fire", 3: "star", 4: "electrocuted"}

SPEED_ITEM_NAMES = {0: "none", 7: "mushroom", 8: "star"}

# eGameState enum from cGame+0x24 (v6+)
GAME_PHASE_NAMES = {
    0: "Pre-match",
    1: "Kickoff",
    2: "Goal celebration",
    3: "Transition",
    4: "Active play",
    5: "Active play (2)",
}

def game_phase_name(phase):
    return GAME_PHASE_NAMES.get(phase, f"Unknown({phase})")

# GCPad button flags
PAD_BUTTONS = {
    0x0001: "LEFT", 0x0002: "RIGHT", 0x0004: "DOWN", 0x0008: "UP",
    0x0010: "Z", 0x0100: "A", 0x0200: "B", 0x0400: "X", 0x0800: "Y",
    0x1000: "START", 0x0020: "L", 0x0040: "R",
}

# -- Action state maps --------------------------------------------------------

STRIKER_ACTION_STATES = {
    0x00: "Deke",
    0x01: "Electrocuted (wall)",
    0x02: "Defensive hit attempt",
    0x03: "Hit by opponent",
    0x04: "Idle turn",
    0x05: "Air shot",
    0x06: "Receiving pass (toad-like)",
    0x08: "Receiving perfect pass",
    0x0A: "Initiating pass",
    0x0C: "Receiving pass",
    0x0D: "Default/idle",
    0x0E: "Ball carrier",
    0x0F: "Offensive sprint",
    0x11: "Ground shot (or chip)",
    0x13: "Defensive slide",
    0x14: "Bumped/body-checked",
    0x15: "Exploding (item hit)",
    0x16: "Hit by item (thrown)",
    0x17: "Slipped on banana",
    0x19: "Hit by big shell",
    0x1A: "Captain slide (no ball)",
    0x1B: "Pre-perfect-pass (niche)",
    0xFF: "State clearing",
}

GOALIE_ACTION_STATES = {
    0x00: "Default/idle",
    0x01: "Walking with ball",
    0x02: "Hands out (preparing)",
    0x03: "Preparing for charged shot",
    0x04: "Jumping/diving to ball",
    0x06: "Laying down (after contact)",
    0x09: "Stunned by ball",
    0x0C: "Throwing ball",
    0x11: "Jogging to ball",
    0x13: "Holding ball",
    0x15: "Walking to pick up ball",
    0x17: "Laying down (goal allowed)",
}


def action_state_name(slot_idx, state):
    """Get human-readable action state name. Slots 4 and 9 are goalies."""
    if slot_idx == 4 or slot_idx == 9:
        return GOALIE_ACTION_STATES.get(state, f"Unknown(0x{state:02X})")
    return STRIKER_ACTION_STATES.get(state, f"Unknown(0x{state:02X})")


# -- Shot classification (ported from matchSettings.js) ------------------------

def classify_shot(ball_pos_x, ball_pos_y, ball_pos_z,
                  ball_vel_x, ball_vel_y, ball_vel_z,
                  charge_amount, prev_shot=None, timestamp=0.0):
    """
    Classify a shot into generic and specific types.
    Returns (generic_type, specific_type).
    Ported from transformShotObject / convertMetricsToGenericShotType /
    convertGenericShotToSpecificShot in matchSettings.js.
    """
    xy_vel = math.sqrt(ball_vel_x**2 + ball_vel_y**2)

    # Generic classification
    if ball_pos_z > 1.25:
        generic = "Air"
    elif xy_vel > 20.5:
        generic = "Ground"
    else:
        generic = "Chip"

    # Specific classification
    specific = "Other"

    # Check if this is a European rebound (prev shot was Dirty European miss)
    if prev_shot is not None:
        prev_specific, prev_made, prev_time = prev_shot
        if prev_specific == "Dirty European" and not prev_made:
            if generic in ("Chip", "Ground"):
                if (timestamp - prev_time <= 5.0 and
                    abs(ball_pos_x) >= 12 and
                    -9 <= ball_pos_y <= 9):
                    return generic, "European"

    if generic == "Chip":
        # Dirty European: chip from midfield sideline area
        if (-3.5 <= ball_pos_x <= 3.5) and (ball_pos_y >= 8 or ball_pos_y <= -8):
            specific = "Dirty European"
        # Box Chip: chip from goalie box corner
        elif (11.5 <= abs(ball_pos_x) <= 14.5) and (-6.5 <= ball_pos_y <= 6.5):
            specific = "Box Chip"

    elif generic == "Ground":
        # Corner: charged ground shot from midfield sideline
        if ((-4.5 <= ball_pos_x <= 4.5) and
            (ball_pos_y >= 8 or ball_pos_y <= -8) and
            charge_amount >= 28):
            specific = "Corner"
        # Slide: charged shot from within the goalie box
        elif ((13.75 <= abs(ball_pos_x) <= 16.5) and
              (-6.5 <= ball_pos_y <= 6.5) and
              charge_amount >= 12):
            specific = "Slide"

    elif generic == "Air":
        specific = "LAB"

    return generic, specific


# -- Character slot names ------------------------------------------------------

def char_slot_name(idx):
    if idx < 4:
        return f"L_striker{idx}"
    elif idx == 4:
        return "L_goalie"
    elif idx < 9:
        return f"R_striker{idx-5}"
    else:
        return "R_goalie"


# -- Data classes --------------------------------------------------------------

@dataclass
class CaptureHeader:
    magic: str
    version: int
    frame_count: int
    fixed_frame_size: int
    left_captain: int
    right_captain: int
    left_sidekick: int
    right_sidekick: int
    stadium_id: int
    # v7+ field geometry
    goal_line_x: float = 0.0
    sideline_y: float = 0.0
    penalty_box_x: float = 0.0
    net_half_width: float = 0.0
    net_height: float = 0.0
    net_depth: float = 0.0

@dataclass
class FrameControllerInput:
    buttons: int
    stick_x: int
    stick_y: int
    substick_x: int
    substick_y: int
    trigger_left: int
    trigger_right: int
    is_connected: int

@dataclass
class FrameCharacter:
    pos_x: float
    pos_y: float
    pos_z: float
    action_state: int
    heading: int
    effect_type: int
    speed_item_type: int
    speed_item_count: int
    is_user_controlled: int
    speed_item_timer: float

@dataclass
class FrameItem:
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    powerup_type: int
    strength_level: int
    slot_index: int
    padding: int
    thrower_pointer: int
    lifetime_timer: int
    target_team_pointer: int
    speed_multiplier: float
    random_id: int
    padding2: int

@dataclass
class FramePowerupSlot:
    type: int  # s32
    charge_count: int
    is_new: int

@dataclass
class FrameTeamStats:
    shots: int
    hits: int
    steals: int
    super_strikes: int
    perfect_passes: int

@dataclass
class GameStateFrame:
    # metadata
    game_time: float
    movie_frame: int
    # score
    left_score: int
    right_score: int
    is_paused: int
    game_phase: int  # eGameState: 0=pre-match, 1=kickoff, 2=celebration, 3=transition, 4/5=active
    # ball
    ball_pos_x: float
    ball_pos_y: float
    ball_pos_z: float
    ball_vel_x: float
    ball_vel_y: float
    ball_vel_z: float
    ball_owner_ptr: int        # who physically has the ball (null while in flight)
    potential_scorer_ptr: int  # v9+: cGame+0x2C — who the game credits for the next goal (0 if v8)
    ball_pass_target_ptr: int  # v10+: ball_ptr+0x30 — pass target character pointer (0 if no active pass)
    is_perfect_pass: int
    ball_charge_amount: float
    # characters (10)
    characters: List[FrameCharacter]
    # character pointers (10)
    character_pointers: List[int]
    # controllers (4)
    controllers: List[FrameControllerInput]
    # inventory (2 teams x 2 slots)
    left_inventory: List[FramePowerupSlot]
    right_inventory: List[FramePowerupSlot]
    # team stats (v8+)
    left_stats: Optional[FrameTeamStats]
    right_stats: Optional[FrameTeamStats]
    # variable items
    item_count: int
    items: List[FrameItem]


# -- Parsing -------------------------------------------------------------------

HEADER_FMT = "<4sIII5B3x6f"  # 48 bytes (v7+: 6 geometry floats after match info)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# v5 FrameCharacter: 28 bytes
# float posX(4) + float posY(4) + float posZ(4) + u32 actionState(4) +
# u16 heading(2) + u8 effectType(1) + u8 speedItemType(1) + u8 speedItemCount(1) +
# u8 isUserControlled(1) + u8 pad[2](2) + float speedItemTimer(4) = 28
CHAR_SIZE = 28

CTRL_SIZE = 10
INVENTORY_SIZE = 8
ITEM_SIZE = 48


def parse_header(data: bytes) -> CaptureHeader:
    vals = struct.unpack_from(HEADER_FMT, data, 0)
    return CaptureHeader(
        magic=vals[0].decode('ascii'),
        version=vals[1],
        frame_count=vals[2],
        fixed_frame_size=vals[3],
        left_captain=vals[4],
        right_captain=vals[5],
        left_sidekick=vals[6],
        right_sidekick=vals[7],
        stadium_id=vals[8],
        goal_line_x=vals[9],
        sideline_y=vals[10],
        penalty_box_x=vals[11],
        net_half_width=vals[12],
        net_height=vals[13],
        net_depth=vals[14],
    )


def parse_character(data: bytes, offset: int) -> FrameCharacter:
    pos_x, pos_y, pos_z = struct.unpack_from("<fff", data, offset)
    action_state = struct.unpack_from("<I", data, offset + 12)[0]
    heading = struct.unpack_from("<H", data, offset + 16)[0]
    effect_type = data[offset + 18]
    speed_item_type = data[offset + 19]
    speed_item_count = data[offset + 20]
    is_user_controlled = data[offset + 21]
    speed_item_timer = struct.unpack_from("<f", data, offset + 24)[0]
    return FrameCharacter(pos_x, pos_y, pos_z, action_state, heading, effect_type,
                          speed_item_type, speed_item_count, is_user_controlled,
                          speed_item_timer)


def parse_controller(data: bytes, offset: int) -> FrameControllerInput:
    buttons = struct.unpack_from("<H", data, offset)[0]
    return FrameControllerInput(
        buttons=buttons,
        stick_x=data[offset + 2],
        stick_y=data[offset + 3],
        substick_x=data[offset + 4],
        substick_y=data[offset + 5],
        trigger_left=data[offset + 6],
        trigger_right=data[offset + 7],
        is_connected=data[offset + 8],
    )


def parse_inventory_slot(data: bytes, offset: int) -> FramePowerupSlot:
    type_val = struct.unpack_from("<i", data, offset)[0]
    charge = data[offset + 4]
    is_new = data[offset + 5]
    return FramePowerupSlot(type_val, charge, is_new)


def parse_item(data: bytes, offset: int) -> FrameItem:
    vals = struct.unpack_from("<ffffffBBBBIIIfHH", data, offset)
    return FrameItem(*vals)


def parse_frame(data: bytes, offset: int, fixed_size: int):
    """Returns (GameStateFrame, bytes_consumed). Supports v5-v10 layouts."""
    o = offset

    game_time, movie_frame = struct.unpack_from("<fI", data, o); o += 8
    left_score, right_score, is_paused, game_phase = struct.unpack_from("<BBBB", data, o); o += 4

    # Ball state
    # v8: 36 bytes (24 pos/vel + 4 owner + 1 perfectPass + 3 pad + 4 charge)
    # v9: 40 bytes (adds 4-byte potentialScorerPtr after owner)
    # v10: 44 bytes (adds 4-byte ballPassTargetPtr after potentialScorerPtr)
    bpx, bpy, bpz, bvx, bvy, bvz = struct.unpack_from("<ffffff", data, o); o += 24
    ball_owner = struct.unpack_from("<I", data, o)[0]; o += 4
    potential_scorer_ptr = 0
    if fixed_size >= 472:  # v9+
        potential_scorer_ptr = struct.unpack_from("<I", data, o)[0]; o += 4
    ball_pass_target_ptr = 0
    if fixed_size >= 476:  # v10+
        ball_pass_target_ptr = struct.unpack_from("<I", data, o)[0]; o += 4
    is_perfect_pass = data[o]; o += 4  # 1 byte + 3 padding
    ball_charge = struct.unpack_from("<f", data, o)[0]; o += 4

    # Characters (10 x 28 = 280)
    chars = []
    for _ in range(10):
        chars.append(parse_character(data, o))
        o += CHAR_SIZE

    # Character pointers (10 x 4 = 40)
    char_ptrs = []
    for _ in range(10):
        char_ptrs.append(struct.unpack_from("<I", data, o)[0])
        o += 4

    # Controllers (4 x 10 = 40)
    ctrls = []
    for _ in range(4):
        ctrls.append(parse_controller(data, o))
        o += CTRL_SIZE

    # Inventory (4 x 8 = 32)
    left_inv = [parse_inventory_slot(data, o), parse_inventory_slot(data, o + INVENTORY_SIZE)]
    o += 2 * INVENTORY_SIZE
    right_inv = [parse_inventory_slot(data, o), parse_inventory_slot(data, o + INVENTORY_SIZE)]
    o += 2 * INVENTORY_SIZE

    # Team stats (v8+: 24 bytes = 5H left + 5H right + 4x padding)
    left_stats = None
    right_stats = None
    if fixed_size >= 468:
        ls = struct.unpack_from("<5H", data, o); o += 10
        rs = struct.unpack_from("<5H", data, o); o += 10
        o += 4  # statsPadding
        left_stats = FrameTeamStats(*ls)
        right_stats = FrameTeamStats(*rs)

    # Item count + padding (4 bytes)
    item_count = data[o]; o += 4

    # Variable items
    items = []
    for _ in range(item_count):
        items.append(parse_item(data, o))
        o += ITEM_SIZE

    frame = GameStateFrame(
        game_time=game_time, movie_frame=movie_frame,
        left_score=left_score, right_score=right_score,
        is_paused=is_paused, game_phase=game_phase,
        ball_pos_x=bpx, ball_pos_y=bpy, ball_pos_z=bpz,
        ball_vel_x=bvx, ball_vel_y=bvy, ball_vel_z=bvz,
        ball_owner_ptr=ball_owner, potential_scorer_ptr=potential_scorer_ptr,
        ball_pass_target_ptr=ball_pass_target_ptr,
        is_perfect_pass=is_perfect_pass,
        ball_charge_amount=ball_charge,
        characters=chars, character_pointers=char_ptrs,
        controllers=ctrls,
        left_inventory=left_inv, right_inventory=right_inv,
        left_stats=left_stats, right_stats=right_stats,
        item_count=item_count, items=items,
    )
    return frame, o - offset


# -- Helpers -------------------------------------------------------------------

def buttons_str(btn):
    parts = []
    for mask, name in PAD_BUTTONS.items():
        if btn & mask:
            parts.append(name)
    return "+".join(parts) if parts else "none"


def heading_degrees(raw):
    return (raw / 65536.0) * 360.0


def ball_speed(f):
    return math.sqrt(f.ball_vel_x**2 + f.ball_vel_y**2 + f.ball_vel_z**2)


def ball_xy_speed(f):
    return math.sqrt(f.ball_vel_x**2 + f.ball_vel_y**2)


def resolve_owner(frame, owner_ptr):
    """Resolve a ball owner pointer to a slot index using character_pointers."""
    if owner_ptr == 0:
        return None, "NONE"
    for ci, ptr in enumerate(frame.character_pointers):
        if ptr == owner_ptr:
            return ci, char_slot_name(ci)
    return None, f"UNRESOLVED(0x{owner_ptr:08X})"


def slot_team(slot_idx):
    """Return 'left' or 'right' for a slot index."""
    return "left" if slot_idx < 5 else "right"


# -- Shot event detection ------------------------------------------------------

def _identify_shooter(frames, frame_idx, team):
    """
    Identify the shooter at the given frame on the given team.
    1. v9+: use potentialScorerPtr (cGame+0x2C) — the game's own authoritative answer.
       This correctly attributes perfect-pass goals where owner=NONE during ball flight.
    2. v8 fallback: scan action states 0x05/0x11/0x08 in a small window around the shot frame.
    3. Last resort: nearest character on the shooting team to the ball.
    Returns (slot_idx, slot_name).
    """
    f = frames[frame_idx]

    # v9+: game's authoritative potential scorer (set on pickup AND shot fire)
    if f.potential_scorer_ptr != 0:
        slot, name = resolve_owner(f, f.potential_scorer_ptr)
        if slot is not None:
            return slot, name

    # Determine which slots belong to this team
    if team == "left":
        striker_slots = [0, 1, 2, 3]
    else:
        striker_slots = [5, 6, 7, 8]

    # v8 fallback: check shooting action states in a small window around the shot frame
    SHOT_STATES = {0x05, 0x11}  # Air shot, Ground shot
    PP_STATES = {0x08}          # Receiving perfect pass
    for window_offset in range(0, 6):
        for direction in (0, -1):
            j = frame_idx + window_offset * (1 if direction == 0 else -1)
            if j < 0 or j >= len(frames):
                continue
            fj = frames[j]
            for si in striker_slots:
                st = fj.characters[si].action_state
                if st in SHOT_STATES or st in PP_STATES:
                    return si, char_slot_name(si)

    # Last resort: nearest character on the shooting team to the ball
    min_dist = 9999
    best_slot = striker_slots[0]
    for si in striker_slots:
        c = f.characters[si]
        dx = c.pos_x - f.ball_pos_x
        dy = c.pos_y - f.ball_pos_y
        d = math.sqrt(dx * dx + dy * dy)
        if d < min_dist:
            min_dist = d
            best_slot = si
    return best_slot, char_slot_name(best_slot)


def detect_shots_v8(frames):
    """
    Detect shots using v8 game stats counters (ground truth from game RAM).
    A shot is detected when leftStats.shots or rightStats.shots increments.
    Shooter is identified via action states + proximity.
    Goal detection uses score counter increments.
    Returns list of dicts with shot details.
    """
    shots = []

    for i in range(1, len(frames)):
        f = frames[i]
        fp = frames[i - 1]

        left_inc = f.left_stats.shots - fp.left_stats.shots
        right_inc = f.right_stats.shots - fp.right_stats.shots

        for team, inc in [("left", left_inc), ("right", right_inc)]:
            if inc <= 0:
                continue

            shooter_slot, shooter_name = _identify_shooter(frames, i, team)

            # Use velocity from one frame after the shot counter increments
            vel_frame_idx = min(i + 1, len(frames) - 1)
            fv = frames[vel_frame_idx]

            # Perfect pass: check shooter action state 0x08 or ball is_perfect_pass
            is_pp = f.is_perfect_pass or fp.is_perfect_pass
            if shooter_slot is not None:
                shooter_state = f.characters[shooter_slot].action_state
                if shooter_state == 0x08:
                    is_pp = True

            # Determine if goal (score changes within next 120 frames)
            pre_left = fp.left_score
            pre_right = fp.right_score
            made = False
            for j in range(i, min(i + 120, len(frames))):
                if (frames[j].left_score != pre_left or
                    frames[j].right_score != pre_right):
                    made = True
                    break

            shots.append({
                'frame': i,
                'game_time': f.game_time,
                'ball_pos': (f.ball_pos_x, f.ball_pos_y, f.ball_pos_z),
                'ball_vel': (fv.ball_vel_x, fv.ball_vel_y, fv.ball_vel_z),
                'xy_speed': ball_xy_speed(fv),
                'total_speed': ball_speed(fv),
                'charge_amount': fp.ball_charge_amount,
                'shooter_slot': shooter_slot,
                'shooter_name': shooter_name,
                'shooter_team': team,
                'made': made,
                'is_perfect_pass': is_pp,
            })

    shots.sort(key=lambda s: s['frame'])
    return shots


def detect_shots_legacy(frames):
    """
    Legacy shot detection for pre-v8 files using velocity-spike heuristics.
    """
    shots = []
    MIN_SHOT_SPEED = 15.0
    COOLDOWN_FRAMES = 30
    last_shot_frame = -COOLDOWN_FRAMES

    for i in range(1, len(frames)):
        f = frames[i]
        fp = frames[i - 1]

        spd = ball_xy_speed(f)
        prev_spd = ball_xy_speed(fp)

        if spd >= MIN_SHOT_SPEED and spd > prev_spd * 1.5 and (i - last_shot_frame) >= COOLDOWN_FRAMES:
            shooter_slot = None
            shooter_name = "Unknown"
            for j in range(i, max(i - 6, 0), -1):
                slot, name = resolve_owner(frames[j], frames[j].ball_owner_ptr)
                if slot is not None:
                    shooter_slot = slot
                    shooter_name = name
                    break

            if shooter_slot in (4, 9):
                continue

            pre_left = fp.left_score
            pre_right = fp.right_score
            made = False
            for j in range(i, min(i + 120, len(frames))):
                if (frames[j].left_score != pre_left or
                    frames[j].right_score != pre_right):
                    made = True
                    break

            shots.append({
                'frame': i,
                'game_time': f.game_time,
                'ball_pos': (f.ball_pos_x, f.ball_pos_y, f.ball_pos_z),
                'ball_vel': (f.ball_vel_x, f.ball_vel_y, f.ball_vel_z),
                'xy_speed': spd,
                'total_speed': ball_speed(f),
                'charge_amount': fp.ball_charge_amount,
                'shooter_slot': shooter_slot,
                'shooter_name': shooter_name,
                'shooter_team': slot_team(shooter_slot) if shooter_slot is not None else "unknown",
                'made': made,
                'is_perfect_pass': fp.is_perfect_pass,
            })
            last_shot_frame = i

    # Fallback: detect goals with no velocity-spike shot
    score_change_frames = []
    for i in range(1, len(frames)):
        if (frames[i].left_score != frames[i - 1].left_score or
            frames[i].right_score != frames[i - 1].right_score):
            score_change_frames.append(i)

    for goal_frame in score_change_frames:
        already_covered = any(s['made'] and abs(s['frame'] - goal_frame) < 120 for s in shots)
        if already_covered:
            continue

        shooter_slot = None
        shooter_name = "Unknown"
        release_frame = goal_frame
        for j in range(goal_frame - 1, max(goal_frame - 180, 0), -1):
            slot, name = resolve_owner(frames[j], frames[j].ball_owner_ptr)
            if slot is not None and slot not in (4, 9):
                shooter_slot = slot
                shooter_name = name
                for k in range(j, min(j + 60, goal_frame)):
                    k_slot, _ = resolve_owner(frames[k], frames[k].ball_owner_ptr)
                    if k_slot != slot:
                        release_frame = k
                        break
                break

        vel_frame_idx = min(release_frame + 1, len(frames) - 1)
        f = frames[release_frame]
        fv = frames[vel_frame_idx]
        fp = frames[max(release_frame - 1, 0)]
        shots.append({
            'frame': release_frame,
            'game_time': f.game_time,
            'ball_pos': (f.ball_pos_x, f.ball_pos_y, f.ball_pos_z),
            'ball_vel': (fv.ball_vel_x, fv.ball_vel_y, fv.ball_vel_z),
            'xy_speed': ball_xy_speed(fv),
            'total_speed': ball_speed(fv),
            'charge_amount': fp.ball_charge_amount,
            'shooter_slot': shooter_slot,
            'shooter_name': shooter_name,
            'shooter_team': slot_team(shooter_slot) if shooter_slot is not None else "unknown",
            'made': True,
            'is_perfect_pass': fp.is_perfect_pass,
        })

    shots.sort(key=lambda s: s['frame'])
    return shots


def detect_shots(frames):
    """Dispatch to v8 counter-based or legacy velocity-spike detection."""
    has_stats = frames[0].left_stats is not None
    if has_stats:
        return detect_shots_v8(frames)
    return detect_shots_legacy(frames)


# -- Main analysis -------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Brian\Documents\Dolphin Emulator\Citrus Replays\output.citframes"

    with open(path, "rb") as f:
        data = f.read()

    header = parse_header(data)
    print("=" * 90)
    print(f"CITF v{header.version} FILE ANALYSIS")
    print("=" * 90)
    print(f"  Magic: {header.magic}  Version: {header.version}")
    print(f"  Frames: {header.frame_count}  Fixed frame size: {header.fixed_frame_size}")
    print(f"  Left:  {CAPTAINS.get(header.left_captain, '?')} + {SIDEKICKS.get(header.left_sidekick, '?')}")
    print(f"  Right: {CAPTAINS.get(header.right_captain, '?')} + {SIDEKICKS.get(header.right_sidekick, '?')}")
    print(f"  Stadium: {STADIUMS.get(header.stadium_id, '?')} (ID {header.stadium_id})")
    print(f"  File size: {len(data):,} bytes")
    print(f"  Field geometry:")
    print(f"    goalLineX={header.goal_line_x:.1f}  sidelineY={header.sideline_y:.1f}  penaltyBoxX={header.penalty_box_x:.1f}")
    print(f"    net: halfWidth={header.net_half_width:.1f}  height={header.net_height:.1f}  depth={header.net_depth:.1f}")

    # Parse all frames
    frames = []
    offset = HEADER_SIZE
    for i in range(header.frame_count):
        frame, consumed = parse_frame(data, offset, header.fixed_frame_size)
        frames.append(frame)
        offset += consumed

    print(f"  Parsed {len(frames)} frames ({offset:,} / {len(data):,} bytes)")
    if offset != len(data):
        print(f"  WARNING: {len(data) - offset} trailing bytes!")
    print()

    # ===========================================================================
    # 1. SHOT DETECTION & CLASSIFICATION
    # ===========================================================================
    print("=" * 90)
    print("SHOT DETECTION & CLASSIFICATION")
    print("=" * 90)

    shots = detect_shots(frames)

    # Classify each shot
    prev_shot_info = None
    for shot in shots:
        bx, by, bz = shot['ball_pos']
        vx, vy, vz = shot['ball_vel']
        generic, specific = classify_shot(
            bx, by, bz, vx, vy, vz,
            shot['charge_amount'],
            prev_shot=prev_shot_info,
            timestamp=shot['game_time']
        )
        shot['generic_type'] = generic
        shot['specific_type'] = specific
        prev_shot_info = (specific, shot['made'], shot['game_time'])

    # Print shot table
    made_shots = [s for s in shots if s['made']]
    missed_shots = [s for s in shots if not s['made']]

    print(f"\n  Total shots detected: {len(shots)}")
    print(f"  Goals: {len(made_shots)}  |  Misses: {len(missed_shots)}")
    if shots:
        print(f"  Shot accuracy: {100*len(made_shots)/len(shots):.0f}%")

    print(f"\n  {'#':>3} {'Time':>7} {'Result':>6} {'Shooter':<12} {'Speed':>6} "
          f"{'Charge':>7} {'Generic':<8} {'Specific':<18} {'PP':>3} {'Position'}")
    print(f"  {'-'*3} {'-'*7} {'-'*6} {'-'*12} {'-'*6} {'-'*7} {'-'*8} {'-'*18} {'-'*3} {'-'*20}")

    for idx, shot in enumerate(shots):
        bx, by, bz = shot['ball_pos']
        result = "GOAL" if shot['made'] else "Miss"
        pp = "Yes" if shot['is_perfect_pass'] else ""
        print(f"  {idx+1:3d} {shot['game_time']:6.1f}s {result:>6} {shot['shooter_name']:<12} "
              f"{shot['total_speed']:5.1f} {shot['charge_amount']:6.1f} "
              f"{shot['generic_type']:<8} {shot['specific_type']:<18} {pp:>3} "
              f"({bx:5.1f},{by:5.1f},{bz:4.1f})")

    # Shot type summary per team
    for team in ("left", "right"):
        team_shots = [s for s in shots if s['shooter_team'] == team]
        if not team_shots:
            continue
        print(f"\n  {team.upper()} TEAM shot type breakdown:")
        type_stats = {}
        for s in team_shots:
            key = s['specific_type']
            if key not in type_stats:
                type_stats[key] = [0, 0]  # [made, total]
            type_stats[key][1] += 1
            if s['made']:
                type_stats[key][0] += 1
        for stype, (made, total) in sorted(type_stats.items(), key=lambda x: -x[1][1]):
            pct = 100 * made / total if total else 0
            print(f"    {stype:<18}: {made}/{total} ({pct:.0f}%)")

    # ===========================================================================
    # 2. GAME STATS TIMELINE (v8+)
    # ===========================================================================
    has_stats = frames[0].left_stats is not None
    if has_stats:
        print(f"\n{'=' * 90}")
        print("GAME STATS TIMELINE")
        print("=" * 90)

        STAT_NAMES = ["shots", "hits", "steals", "super_strikes", "perfect_passes"]
        STAT_LABELS = {"shots": "Shot", "hits": "Hit", "steals": "Steal",
                       "super_strikes": "Super Strike", "perfect_passes": "Perfect Pass"}

        events = []
        for i in range(1, len(frames)):
            f = frames[i]
            fp = frames[i - 1]
            # Score changes
            if f.left_score != fp.left_score:
                events.append((i, f.game_time, "LEFT", "GOAL",
                               fp.left_score, f.left_score))
            if f.right_score != fp.right_score:
                events.append((i, f.game_time, "RIGHT", "GOAL",
                               fp.right_score, f.right_score))
            # Stat counter changes
            for stat in STAT_NAMES:
                for team, stats, prev_stats in [("LEFT", f.left_stats, fp.left_stats),
                                                ("RIGHT", f.right_stats, fp.right_stats)]:
                    cur = getattr(stats, stat)
                    prev = getattr(prev_stats, stat)
                    if cur > prev:
                        events.append((i, f.game_time, team, STAT_LABELS[stat], prev, cur))

        if events:
            print(f"\n  {'Frame':>7} {'Time':>7} {'Team':<6} {'Event':<15} {'Counter'}")
            print(f"  {'-'*7} {'-'*7} {'-'*6} {'-'*15} {'-'*10}")
            for fi, t, team, label, prev_val, cur_val in events:
                print(f"  {fi:>7} {t:6.2f}s {team:<6} {label:<15} {prev_val} -> {cur_val}")
        else:
            print("\n  No stat counter changes detected.")

        # Final totals
        last = frames[-1]
        print(f"\n  Final stats:")
        print(f"    {'Stat':<16} {'Left':>6} {'Right':>6}")
        print(f"    {'-'*16} {'-'*6} {'-'*6}")
        print(f"    {'Score':<16} {last.left_score:>6} {last.right_score:>6}")
        for stat in STAT_NAMES:
            l_val = getattr(last.left_stats, stat)
            r_val = getattr(last.right_stats, stat)
            print(f"    {STAT_LABELS[stat]:<16} {l_val:>6} {r_val:>6}")

    # ===========================================================================
    # 3. SCORING TIMELINE (detailed per-goal narrative)
    # ===========================================================================
    print(f"\n{'=' * 90}")
    print("SCORING TIMELINE")
    print("=" * 90)

    # Detect score changes
    score_events = []
    prev_left = 0
    prev_right = 0
    for i, f in enumerate(frames):
        if f.left_score != prev_left or f.right_score != prev_right:
            side = "LEFT" if f.left_score > prev_left else "RIGHT"
            delta = (f.left_score - prev_left) if side == "LEFT" else (f.right_score - prev_right)
            score_events.append((i, f, side, delta))
            prev_left = f.left_score
            prev_right = f.right_score

    if not score_events:
        print("  No goals scored in this match!")
    else:
        for event_idx, (fi, sf, side, delta) in enumerate(score_events):
            print(f"\n{'-' * 80}")
            print(f"  GOAL #{event_idx+1}: {side} scores +{delta}  ->  "
                  f"Left {sf.left_score} - {sf.right_score} Right")
            print(f"  Frame {fi} | Game time: {sf.game_time:.2f}s | Movie frame: {sf.movie_frame}")

            # Find the matching shot event
            matching_shot = None
            for shot in made_shots:
                if abs(shot['frame'] - fi) < 120:
                    matching_shot = shot
                    break

            if matching_shot:
                print(f"  Shot type: {matching_shot['specific_type']} ({matching_shot['generic_type']})")
                print(f"  Shooter: {matching_shot['shooter_name']}")
                print(f"  Ball speed: {matching_shot['total_speed']:.1f} (XY: {matching_shot['xy_speed']:.1f})")
                print(f"  Charge amount: {matching_shot['charge_amount']:.1f}")
                if matching_shot['is_perfect_pass']:
                    print(f"  PERFECT PASS active at shot time!")

            # Analyze the 180 frames (~3 seconds) leading up to the goal
            lookback = 180
            start = max(0, fi - lookback)

            # Ball trajectory leading to goal
            max_speed = 0
            shot_frame = None
            for j in range(start, fi):
                spd = ball_speed(frames[j])
                if spd > max_speed:
                    max_speed = spd
                    shot_frame = j

            if shot_frame is not None:
                sf2 = frames[shot_frame]
                print(f"\n  Peak ball speed: {max_speed:.2f} at frame {shot_frame} "
                      f"(game time {sf2.game_time:.2f}s)")
                print(f"  Ball pos at peak: ({sf2.ball_pos_x:.1f}, {sf2.ball_pos_y:.1f}, {sf2.ball_pos_z:.1f})")
                print(f"  Ball vel at peak: ({sf2.ball_vel_x:.1f}, {sf2.ball_vel_y:.1f}, {sf2.ball_vel_z:.1f})")
                owner_slot, owner_name = resolve_owner(sf2, sf2.ball_owner_ptr)
                if sf2.ball_owner_ptr:
                    print(f"  Ball owner: {owner_name}")
                if sf2.ball_charge_amount > 0:
                    print(f"  Charge at peak: {sf2.ball_charge_amount:.1f}")

            # Ball ownership changes in leadup
            print(f"\n  Ball ownership changes (last ~3s):")
            prev_owner = frames[start].ball_owner_ptr if start > 0 else 0
            for j in range(start, fi):
                f2 = frames[j]
                if f2.ball_owner_ptr != prev_owner:
                    _, owner_name = resolve_owner(f2, f2.ball_owner_ptr)
                    print(f"    Frame {j} (t={f2.game_time:.2f}s): -> {owner_name}"
                          f"  ball@({f2.ball_pos_x:.1f},{f2.ball_pos_y:.1f})")
                    prev_owner = f2.ball_owner_ptr

            # Character action states near the shot
            if shot_frame is not None:
                sf2 = frames[shot_frame]
                print(f"\n  Character states at shot (frame {shot_frame}):")
                for ci in range(10):
                    c = sf2.characters[ci]
                    state_name = action_state_name(ci, c.action_state)
                    ctrl_flag = " [HUMAN]" if c.is_user_controlled else ""
                    z_info = f" z={c.pos_z:.2f}" if c.pos_z > 0.01 else ""
                    print(f"    {char_slot_name(ci):<12}: 0x{c.action_state:02X} {state_name:<25}"
                          f" pos=({c.pos_x:6.1f},{c.pos_y:6.1f}){z_info}{ctrl_flag}")

            # Items active during approach
            items_seen = set()
            for j in range(start, fi):
                for item in frames[j].items:
                    items_seen.add(POWERUP_TYPES.get(item.powerup_type, f"type{item.powerup_type}"))
            if items_seen:
                print(f"\n  Items active during approach: {', '.join(items_seen)}")

            # Character effects during approach
            effects_seen = []
            for j in range(max(0, fi - 60), fi):
                for ci in range(10):
                    eff = frames[j].characters[ci].effect_type
                    if eff != 0:
                        effects_seen.append((j, ci, eff))
            if effects_seen:
                print(f"\n  Character effects near goal:")
                reported = set()
                for ej, eci, eeff in effects_seen:
                    key = (eci, eeff)
                    if key not in reported:
                        print(f"    {char_slot_name(eci)}: {EFFECT_NAMES.get(eeff, '?')} (frame {ej})")
                        reported.add(key)

            # Controller inputs at the shot moment
            if shot_frame is not None:
                print(f"\n  Controller inputs at shot (frame {shot_frame}):")
                for port in range(2):
                    ci = frames[shot_frame].controllers[port]
                    print(f"    P{port+1}: {buttons_str(ci.buttons):20s} "
                          f"stick=({ci.stick_x},{ci.stick_y}) "
                          f"cstick=({ci.substick_x},{ci.substick_y})")

            # Goalie position at moment of goal
            scoring_frame = frames[fi]
            if side == "LEFT":
                gk = scoring_frame.characters[9]
                gk_label = "R_goalie"
            else:
                gk = scoring_frame.characters[4]
                gk_label = "L_goalie"
            gk_state = action_state_name(9 if side == "LEFT" else 4, gk.action_state)
            dist_to_ball = math.sqrt(
                (gk.pos_x - scoring_frame.ball_pos_x)**2 +
                (gk.pos_y - scoring_frame.ball_pos_y)**2)
            print(f"\n  {gk_label} at goal frame: pos=({gk.pos_x:.1f},{gk.pos_y:.1f}) "
                  f"state=0x{gk.action_state:02X} ({gk_state}) "
                  f"dist_to_ball={dist_to_ball:.1f}")

    # ===========================================================================
    # 4. ACTION STATE DISTRIBUTION
    # ===========================================================================
    print(f"\n{'=' * 90}")
    print("ACTION STATE ANALYSIS")
    print("=" * 90)

    # Striker action states (slots 0-3, 5-8)
    striker_counts = {}
    goalie_counts = {}
    for f in frames:
        for ci in range(10):
            st = f.characters[ci].action_state
            if ci == 4 or ci == 9:
                goalie_counts[st] = goalie_counts.get(st, 0) + 1
            else:
                striker_counts[st] = striker_counts.get(st, 0) + 1

    total_striker_frames = len(frames) * 8
    total_goalie_frames = len(frames) * 2

    print(f"\n  STRIKER action states (8 strikers x {len(frames)} frames = {total_striker_frames} samples):")
    print(f"  {'State':>7} {'Name':<30} {'Count':>8} {'%':>6}")
    print(f"  {'-'*7} {'-'*30} {'-'*8} {'-'*6}")
    for st, cnt in sorted(striker_counts.items(), key=lambda x: -x[1]):
        name = STRIKER_ACTION_STATES.get(st, f"Unknown")
        pct = 100 * cnt / total_striker_frames
        if pct >= 0.1:  # only show states with >= 0.1%
            print(f"  0x{st:04X}  {name:<30} {cnt:>8} {pct:5.1f}%")

    print(f"\n  GOALIE action states (2 goalies x {len(frames)} frames = {total_goalie_frames} samples):")
    print(f"  {'State':>7} {'Name':<30} {'Count':>8} {'%':>6}")
    print(f"  {'-'*7} {'-'*30} {'-'*8} {'-'*6}")
    for st, cnt in sorted(goalie_counts.items(), key=lambda x: -x[1]):
        name = GOALIE_ACTION_STATES.get(st, f"Unknown")
        pct = 100 * cnt / total_goalie_frames
        if pct >= 0.1:
            print(f"  0x{st:04X}  {name:<30} {cnt:>8} {pct:5.1f}%")

    # ===========================================================================
    # 5. POSSESSION & CONTROL ANALYSIS
    # ===========================================================================
    print(f"\n{'=' * 90}")
    print("POSSESSION & CONTROL ANALYSIS")
    print("=" * 90)

    left_possession = 0
    right_possession = 0
    loose_frames = 0
    for f in frames:
        slot, _ = resolve_owner(f, f.ball_owner_ptr)
        if slot is None:
            loose_frames += 1
        elif slot < 5:
            left_possession += 1
        else:
            right_possession += 1

    total_owned = left_possession + right_possession
    print(f"\n  Ball possession:")
    print(f"    Left team:  {left_possession:>5} frames ({100*left_possession/len(frames):.1f}%)")
    print(f"    Right team: {right_possession:>5} frames ({100*right_possession/len(frames):.1f}%)")
    print(f"    Loose ball: {loose_frames:>5} frames ({100*loose_frames/len(frames):.1f}%)")
    if total_owned > 0:
        print(f"    Possession split (owned only): Left {100*left_possession/total_owned:.0f}% / Right {100*right_possession/total_owned:.0f}%")

    # Per-slot possession
    slot_possession = [0] * 10
    for f in frames:
        slot, _ = resolve_owner(f, f.ball_owner_ptr)
        if slot is not None:
            slot_possession[slot] += 1

    print(f"\n  Per-character possession:")
    for ci in range(10):
        if slot_possession[ci] > 0:
            pct = 100 * slot_possession[ci] / len(frames)
            bar = "#" * int(pct)
            print(f"    {char_slot_name(ci):<12}: {slot_possession[ci]:>5} frames ({pct:4.1f}%) {bar}")

    # Human control distribution
    print(f"\n  Human-controlled frames per slot:")
    ctrl_frames = [0] * 10
    for f in frames:
        for ci in range(10):
            if f.characters[ci].is_user_controlled:
                ctrl_frames[ci] += 1
    for ci in range(10):
        if ctrl_frames[ci] > 0:
            pct = 100 * ctrl_frames[ci] / len(frames)
            print(f"    {char_slot_name(ci):<12}: {ctrl_frames[ci]:>5} frames ({pct:4.1f}%)")

    # ===========================================================================
    # 6. BALL CHARGE ANALYSIS
    # ===========================================================================
    print(f"\n{'=' * 90}")
    print("BALL CHARGE ANALYSIS")
    print("=" * 90)

    charge_frames = [(i, f.ball_charge_amount) for i, f in enumerate(frames) if f.ball_charge_amount > 0]
    print(f"\n  Frames with non-zero charge: {len(charge_frames)} / {len(frames)} ({100*len(charge_frames)/len(frames):.1f}%)")
    if charge_frames:
        max_charge = max(c for _, c in charge_frames)
        print(f"  Max charge seen: {max_charge:.1f}")

        # Charge distribution buckets
        buckets = {"0-10": 0, "10-20": 0, "20-30": 0, "30-40": 0, "40+": 0}
        for _, c in charge_frames:
            if c < 10: buckets["0-10"] += 1
            elif c < 20: buckets["10-20"] += 1
            elif c < 30: buckets["20-30"] += 1
            elif c < 40: buckets["30-40"] += 1
            else: buckets["40+"] += 1
        print(f"  Charge distribution:")
        for bucket, cnt in buckets.items():
            if cnt > 0:
                print(f"    {bucket:>5}: {cnt:>5} frames")

    # ===========================================================================
    # 7. GAME PHASE ANALYSIS
    # ===========================================================================
    print(f"\n{'=' * 90}")
    print("GAME PHASE ANALYSIS")
    print("=" * 90)

    # Phase distribution from eGameState field (v6+)
    phase_counts = {}
    for f in frames:
        phase_counts[f.game_phase] = phase_counts.get(f.game_phase, 0) + 1

    print(f"\n  eGameState distribution ({len(frames)} frames):")
    print(f"  {'Phase':>5} {'Name':<25} {'Frames':>7} {'%':>6}")
    print(f"  {'-'*5} {'-'*25} {'-'*7} {'-'*6}")
    for phase in sorted(phase_counts.keys()):
        cnt = phase_counts[phase]
        pct = 100 * cnt / len(frames)
        print(f"  {phase:>5} {game_phase_name(phase):<25} {cnt:>7} {pct:5.1f}%")

    # Active play = phase 4 or 5
    active_by_phase = sum(cnt for p, cnt in phase_counts.items() if p in (4, 5))
    non_active = len(frames) - active_by_phase
    print(f"\n  Active play (phase 4/5): {active_by_phase} frames ({100*active_by_phase/len(frames):.1f}%)")
    print(f"  Non-active (phase 0-3):  {non_active} frames ({100*non_active/len(frames):.1f}%)")

    # Cross-tabulate phase with other signals
    print(f"\n  Phase vs. frozen gameTime / perfectPass / isPaused:")
    print(f"  {'Phase':<20} {'Total':>6} {'Frozen':>7} {'PerfPass':>9} {'Paused':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*9} {'-'*7}")
    for phase in sorted(phase_counts.keys()):
        total_p = 0
        frozen_p = 0
        pp_p = 0
        paused_p = 0
        for i, f in enumerate(frames):
            if f.game_phase != phase:
                continue
            total_p += 1
            if i > 0 and f.game_time == frames[i - 1].game_time:
                frozen_p += 1
            if f.is_perfect_pass:
                pp_p += 1
            if f.is_paused:
                paused_p += 1
        print(f"  {game_phase_name(phase):<20} {total_p:>6} {frozen_p:>7} {pp_p:>9} {paused_p:>7}")

    # Frame classification for AI training
    actionable = 0
    pp_actionable = 0
    irrelevant = 0
    for f in frames:
        if f.game_phase in (4, 5) and not f.is_paused:
            if f.is_perfect_pass:
                pp_actionable += 1
            else:
                actionable += 1
        else:
            irrelevant += 1
    print(f"\n  AI training classification:")
    print(f"    Actionable (active + not paused):     {actionable:>6} ({100*actionable/len(frames):.1f}%)")
    print(f"    Actionable + perfect pass:            {pp_actionable:>6} ({100*pp_actionable/len(frames):.1f}%)")
    print(f"    Irrelevant (celebration/kickoff/etc):  {irrelevant:>6} ({100*irrelevant/len(frames):.1f}%)")

    # Phase transitions timeline
    transitions = []
    for i in range(1, len(frames)):
        if frames[i].game_phase != frames[i - 1].game_phase:
            transitions.append((i, frames[i - 1].game_phase, frames[i].game_phase, frames[i].game_time))

    print(f"\n  Phase transitions: {len(transitions)} total")
    if transitions:
        print(f"  {'Frame':>7} {'Time':>7} {'From':<20} {'To':<20}")
        print(f"  {'-'*7} {'-'*7} {'-'*20} {'-'*20}")
        for fi, from_p, to_p, t in transitions:
            print(f"  {fi:>7} {t:6.2f}s {game_phase_name(from_p):<20} {game_phase_name(to_p):<20}")

    # ===========================================================================
    # 8. MATCH SUMMARY
    # ===========================================================================
    print(f"\n{'=' * 90}")
    print("MATCH SUMMARY")
    print("=" * 90)

    last = frames[-1]
    print(f"\n  Final score: Left {last.left_score} - {last.right_score} Right")
    print(f"  Total frames: {len(frames)}")
    print(f"  Game time range: {frames[0].game_time:.2f}s -> {last.game_time:.2f}s")

    paused_frames = sum(1 for f in frames if f.is_paused)
    print(f"  Paused frames: {paused_frames}")

    frames_with_items = sum(1 for f in frames if f.item_count > 0)
    max_items = max(f.item_count for f in frames) if frames else 0
    print(f"  Frames with active items: {frames_with_items} / {len(frames)} "
          f"({100*frames_with_items/len(frames):.1f}%)")
    print(f"  Max simultaneous items: {max_items}")

    pp_frames = sum(1 for f in frames if f.is_perfect_pass)
    print(f"  Frames with perfect pass: {pp_frames}")

    # Ball speed stats
    speeds = [ball_speed(f) for f in frames]
    avg_speed = sum(speeds) / len(speeds)
    print(f"  Ball speed: avg={avg_speed:.2f}, max={max(speeds):.2f}")

    # Character Z stats
    all_z = [c.pos_z for f in frames for c in f.characters]
    nonzero_z = [z for z in all_z if abs(z) > 0.01]
    print(f"  Character Z: min={min(all_z):.3f}, max={max(all_z):.3f}, "
          f"airborne (|z|>0.01): {len(nonzero_z)} / {len(all_z)} ({100*len(nonzero_z)/len(all_z):.1f}%)")

    # Field bounds
    print(f"\n  Spatial bounds:")
    min_bx = min(f.ball_pos_x for f in frames)
    max_bx = max(f.ball_pos_x for f in frames)
    min_by = min(f.ball_pos_y for f in frames)
    max_by = max(f.ball_pos_y for f in frames)
    min_bz = min(f.ball_pos_z for f in frames)
    max_bz = max(f.ball_pos_z for f in frames)
    print(f"    Ball X: [{min_bx:.1f}, {max_bx:.1f}]  Y: [{min_by:.1f}, {max_by:.1f}]  Z: [{min_bz:.1f}, {max_bz:.1f}]")

    all_cx = [c.pos_x for f in frames for c in f.characters]
    all_cy = [c.pos_y for f in frames for c in f.characters]
    print(f"    Char X: [{min(all_cx):.1f}, {max(all_cx):.1f}]  Y: [{min(all_cy):.1f}, {max(all_cy):.1f}]")

    # ===========================================================================
    # 9. DETAILED FRAME-BY-FRAME NEAR GOALS
    # ===========================================================================
    if score_events:
        print(f"\n{'=' * 90}")
        print("DETAILED FRAME-BY-FRAME NEAR GOALS")
        print("=" * 90)

        for event_idx, (fi, sf, side, delta) in enumerate(score_events):
            print(f"\n{'-' * 80}")
            print(f"GOAL #{event_idx+1} -- 60-frame leadup")

            start = max(0, fi - 60)
            for j in range(start, min(fi + 5, len(frames)), 3):
                f = frames[j]
                spd = ball_speed(f)
                _, owner_name = resolve_owner(f, f.ball_owner_ptr)

                # Find closest character to ball
                min_dist = 9999
                closest = -1
                for ci in range(10):
                    c = f.characters[ci]
                    dx = c.pos_x - f.ball_pos_x
                    dy = c.pos_y - f.ball_pos_y
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < min_dist:
                        min_dist = d
                        closest = ci

                _, scorer_name = resolve_owner(f, f.potential_scorer_ptr)
                scorer_str = f" | scorer={scorer_name:<12}" if f.potential_scorer_ptr != 0 else ""
                marker = " ***GOAL" if j == fi else ""
                charge_str = f" chg={f.ball_charge_amount:.0f}" if f.ball_charge_amount > 0 else ""
                print(f"  F{j:5d} t={f.game_time:6.2f}s | "
                      f"ball=({f.ball_pos_x:7.1f},{f.ball_pos_y:7.1f},{f.ball_pos_z:5.1f}) "
                      f"spd={spd:5.1f}{charge_str} | owner={owner_name:<12}{scorer_str} | "
                      f"nearest={char_slot_name(closest)}({min_dist:.1f}) | "
                      f"score={f.left_score}-{f.right_score}{marker}")

    print(f"\n{'=' * 90}")
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
