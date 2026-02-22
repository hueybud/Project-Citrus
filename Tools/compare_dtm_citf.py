#!/usr/bin/env python3
"""
Compare controller 1 inputs between DTM and CITF files.

IMPORTANT NOTES:
1. CITF Frame Offset: CITF frames are offset by ~2 frames from DTM due to when
   BeginCapture() is called. CITF Frame 0 typically corresponds to DTM Frame 2.
   Check the movieFrameNumber field in CITF frames for exact DTM frame mapping.

2. Button Format Difference: DTM uses ControllerState bit fields, CITF uses
   PAD_BUTTON_* flags. Script converts DTM format to PAD_BUTTON_* for comparison.

3. Data Integrity: The state-action pairs in CITF are correct. Each frame's
   controller inputs are the ones that produced that frame's game state.
"""

import struct
import sys
from pathlib import Path

ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'

def load_citf_bytes(path):
    """Read a .citframes file, decompressing with zstd if needed."""
    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:4] == ZSTD_MAGIC:
        try:
            import zstandard as zstd
        except ImportError:
            sys.exit("zstandard package required for compressed CITF files: pip install zstandard")
        return zstd.ZstdDecompressor().decompress(raw)
    return raw

# PAD_BUTTON_* and PAD_* flag definitions (from GCPadStatus.h)
PAD_BUTTON_LEFT = 0x0001
PAD_BUTTON_RIGHT = 0x0002
PAD_BUTTON_DOWN = 0x0004
PAD_BUTTON_UP = 0x0008
PAD_TRIGGER_Z = 0x0010
PAD_TRIGGER_R = 0x0020
PAD_TRIGGER_L = 0x0040
PAD_USE_ORIGIN = 0x0080
PAD_BUTTON_A = 0x0100
PAD_BUTTON_B = 0x0200
PAD_BUTTON_X = 0x0400
PAD_BUTTON_Y = 0x0800
PAD_BUTTON_START = 0x1000
PAD_GET_ORIGIN = 0x2000

def convert_dtm_buttons_to_gcpad(dtm_buttons):
    """
    Convert DTM ControllerState button bits to GCPadStatus PAD_BUTTON_* format.

    ControllerState bit layout (16 bits):
    Bit 0: Start
    Bit 1: A
    Bit 2: B
    Bit 3: X
    Bit 4: Y
    Bit 5: Z
    Bit 6: DPadUp
    Bit 7: DPadDown
    Bit 8: DPadLeft
    Bit 9: DPadRight
    Bit 10: L
    Bit 11: R
    Bit 12: disc
    Bit 13: reset
    Bit 14: is_connected
    Bit 15: get_origin
    """
    gcpad_buttons = PAD_USE_ORIGIN  # Dolphin always sets this

    # Map ControllerState bits to PAD_BUTTON_* flags
    if dtm_buttons & (1 << 0):  # Start
        gcpad_buttons |= PAD_BUTTON_START
    if dtm_buttons & (1 << 1):  # A
        gcpad_buttons |= PAD_BUTTON_A
    if dtm_buttons & (1 << 2):  # B
        gcpad_buttons |= PAD_BUTTON_B
    if dtm_buttons & (1 << 3):  # X
        gcpad_buttons |= PAD_BUTTON_X
    if dtm_buttons & (1 << 4):  # Y
        gcpad_buttons |= PAD_BUTTON_Y
    if dtm_buttons & (1 << 5):  # Z
        gcpad_buttons |= PAD_TRIGGER_Z
    if dtm_buttons & (1 << 6):  # DPadUp
        gcpad_buttons |= PAD_BUTTON_UP
    if dtm_buttons & (1 << 7):  # DPadDown
        gcpad_buttons |= PAD_BUTTON_DOWN
    if dtm_buttons & (1 << 8):  # DPadLeft
        gcpad_buttons |= PAD_BUTTON_LEFT
    if dtm_buttons & (1 << 9):  # DPadRight
        gcpad_buttons |= PAD_BUTTON_RIGHT
    if dtm_buttons & (1 << 10):  # L
        gcpad_buttons |= PAD_TRIGGER_L
    if dtm_buttons & (1 << 11):  # R
        gcpad_buttons |= PAD_TRIGGER_R
    if dtm_buttons & (1 << 15):  # get_origin
        gcpad_buttons |= PAD_GET_ORIGIN

    return gcpad_buttons

def parse_dtm_header(data):
    """Parse DTM file header."""
    magic = data[0:4]
    if magic != b'DTM\x1A':
        raise ValueError(f"Invalid DTM magic: {magic}")

    num_controllers = data[0x0B]
    frame_count = struct.unpack('<Q', data[0x0D:0x15])[0]
    input_count = struct.unpack('<Q', data[0x15:0x1D])[0]

    # Controller bitmask
    controllers_bitmask = data[0x0B]

    return {
        'num_controllers': num_controllers,
        'frame_count': frame_count,
        'input_count': input_count,
        'controllers_bitmask': controllers_bitmask,
        'header_size': 256
    }

def parse_citf_header(data):
    """Parse CITF file header (v7-v11)."""
    magic = data[0:4]
    if magic != b'CITF':
        raise ValueError(f"Invalid CITF magic: {magic}")

    version = struct.unpack('<I', data[0x04:0x08])[0]
    frame_count = struct.unpack('<I', data[0x08:0x0C])[0]
    fixed_frame_size = struct.unpack('<I', data[0x0C:0x10])[0]

    # v11+ header is 273 bytes (48-byte base + 225-byte match metadata)
    header_size = 273 if version >= 11 else 48

    return {
        'version': version,
        'frame_count': frame_count,
        'fixed_frame_size': fixed_frame_size,
        'header_size': header_size
    }

def parse_dtm_controller_input(data, offset):
    """Parse one controller input from DTM (8 bytes per controller).

    Official DTM format per https://tasvideos.org/EmulatorResources/Dolphin/DTM:
    Bytes 0-1: Button states (ControllerState bit fields)
    Bytes 2-3: L trigger, R trigger (0-255)
    Bytes 4-5: Analog stick X, Y (0-255)
    Bytes 6-7: C-stick X, Y (0-255)
    """
    dtm_buttons = struct.unpack('<H', data[offset:offset+2])[0]
    trigger_left = data[offset+2]
    trigger_right = data[offset+3]
    stick_x = data[offset+4]
    stick_y = data[offset+5]
    substick_x = data[offset+6]
    substick_y = data[offset+7]

    # Convert DTM ControllerState format to GCPadStatus PAD_BUTTON_* format
    gcpad_buttons = convert_dtm_buttons_to_gcpad(dtm_buttons)

    return {
        'buttons': gcpad_buttons,  # Converted to PAD_BUTTON_* format
        'dtm_raw_buttons': dtm_buttons,  # Keep raw for debugging
        'stickX': stick_x,
        'stickY': stick_y,
        'substickX': substick_x,
        'substickY': substick_y,
        'triggerLeft': trigger_left,
        'triggerRight': trigger_right
    }

def parse_citf_frame_metadata(data, frame_offset):
    """Parse CITF frame metadata to get movieFrameNumber."""
    # GameStateFrame layout:
    # Offset 0: gameTime (float, 4 bytes)
    # Offset 4: movieFrameNumber (u32, 4 bytes)
    movie_frame_number = struct.unpack('<I', data[frame_offset+4:frame_offset+8])[0]
    return movie_frame_number

def parse_citf_controller_input(data, offset):
    """Parse one controller input from CITF (10 bytes per controller)."""
    buttons = struct.unpack('<H', data[offset:offset+2])[0]
    stick_x = data[offset+2]
    stick_y = data[offset+3]
    substick_x = data[offset+4]
    substick_y = data[offset+5]
    trigger_left = data[offset+6]
    trigger_right = data[offset+7]
    is_connected = data[offset+8]

    return {
        'buttons': buttons,
        'stickX': stick_x,
        'stickY': stick_y,
        'substickX': substick_x,
        'substickY': substick_y,
        'triggerLeft': trigger_left,
        'triggerRight': trigger_right,
        'isConnected': is_connected
    }

def compare_controller1(dtm_path, citf_path, sample_every=1, show_mismatches=10):
    """Compare controller 1 inputs between DTM and CITF files."""

    # Read both files
    with open(dtm_path, 'rb') as f:
        dtm_data = f.read()

    citf_data = load_citf_bytes(citf_path)

    # Parse headers
    print("=" * 80)
    print("CONTROLLER 1 INPUT COMPARISON")
    print("=" * 80)
    print("\nParsing DTM header...")
    dtm_header = parse_dtm_header(dtm_data)
    print(f"  Frame count: {dtm_header['frame_count']}")
    print(f"  Controller bitmask: 0x{dtm_header['controllers_bitmask']:02X}")
    print(f"  Controller 1 connected: {'Yes' if (dtm_header['controllers_bitmask'] & 0x01) else 'No'}")

    print("\nParsing CITF header...")
    citf_header = parse_citf_header(citf_data)
    print(f"  Frame count: {citf_header['frame_count']}")
    print(f"  Fixed frame size: {citf_header['fixed_frame_size']} bytes")

    # Check frame counts match
    if dtm_header['frame_count'] != citf_header['frame_count']:
        print(f"\n[WARNING] Frame count mismatch!")
        print(f"  DTM: {dtm_header['frame_count']}")
        print(f"  CITF: {citf_header['frame_count']}")

    frames_to_check = min(dtm_header['frame_count'], citf_header['frame_count'])

    print(f"\nComparing {frames_to_check} frames (sampling every {sample_every})...")
    print("=" * 80)

    # DTM: Controller 1 is always first in the input stream (if connected)
    dtm_input_offset = dtm_header['header_size']
    dtm_bytes_per_frame = 8  # Only controller 1 data

    # CITF: Controller 1 is at index 0 in the controllers array
    citf_fixed_size = citf_header['fixed_frame_size']
    citf_controller_offset_in_frame = 284  # Controllers start at byte 284 (updated for movieFrameNumber field)
    citf_item_count_offset = 356  # itemCount u8 at offset 356 within each frame

    # Build frame offset table by sequentially scanning (frames are variable-size)
    citf_frame_offsets = []
    pos = citf_header['header_size']
    for i in range(citf_header['frame_count']):
        citf_frame_offsets.append(pos)
        item_count = citf_data[pos + citf_item_count_offset]
        pos += citf_fixed_size + item_count * 48  # 48 = sizeof(FrameItem)

    mismatches = []
    matches = 0

    # Track statistics
    stats = {
        'button_mismatches': 0,
        'stick_mismatches': 0,
        'substick_mismatches': 0,
        'trigger_mismatches': 0
    }

    for citf_frame_idx in range(0, citf_header['frame_count'], sample_every):
        # Parse CITF frame to get movieFrameNumber
        citf_frame_start = citf_frame_offsets[citf_frame_idx]
        movie_frame_num = parse_citf_frame_metadata(citf_data, citf_frame_start)

        # Skip if corresponding DTM frame doesn't exist
        if movie_frame_num >= dtm_header['frame_count']:
            continue

        # Parse DTM controller 1 input using movieFrameNumber
        dtm_offset = dtm_input_offset + (movie_frame_num * dtm_bytes_per_frame)

        # Debug: Print mapping for first few frames
        if citf_frame_idx < 3:
            raw_bytes = dtm_data[dtm_offset:dtm_offset+8]
            print(f"CITF Frame {citf_frame_idx} -> DTM Frame {movie_frame_num} (bytes: {raw_bytes.hex()})")

        dtm_ctrl = parse_dtm_controller_input(dtm_data, dtm_offset)

        # Parse CITF controller 1 input (port 0)
        citf_ctrl_offset = citf_frame_start + citf_controller_offset_in_frame
        citf_ctrl = parse_citf_controller_input(citf_data, citf_ctrl_offset)

        # Compare all fields
        # NOTE: DTM buttons are converted from ControllerState to PAD_BUTTON_* format
        mismatch_fields = []

        # Compare buttons (after conversion)
        if dtm_ctrl['buttons'] != citf_ctrl['buttons']:
            mismatch_fields.append('buttons')
            stats['button_mismatches'] += 1

        # Tolerate ±1 difference in sticks (sampling jitter)
        stick_x_diff = abs(dtm_ctrl['stickX'] - citf_ctrl['stickX'])
        stick_y_diff = abs(dtm_ctrl['stickY'] - citf_ctrl['stickY'])
        if stick_x_diff > 1 or stick_y_diff > 1:
            mismatch_fields.append('stick')
            stats['stick_mismatches'] += 1

        # Tolerate ±1 difference in c-sticks (sampling jitter)
        substick_x_diff = abs(dtm_ctrl['substickX'] - citf_ctrl['substickX'])
        substick_y_diff = abs(dtm_ctrl['substickY'] - citf_ctrl['substickY'])
        if substick_x_diff > 1 or substick_y_diff > 1:
            mismatch_fields.append('substick')
            stats['substick_mismatches'] += 1

        # Tolerate ±1 difference in triggers (sampling jitter)
        trigger_l_diff = abs(dtm_ctrl['triggerLeft'] - citf_ctrl['triggerLeft'])
        trigger_r_diff = abs(dtm_ctrl['triggerRight'] - citf_ctrl['triggerRight'])
        if trigger_l_diff > 1 or trigger_r_diff > 1:
            mismatch_fields.append('triggers')
            stats['trigger_mismatches'] += 1

        if mismatch_fields:
            mismatches.append({
                'citf_frame': citf_frame_idx,
                'dtm_frame': movie_frame_num,
                'fields': mismatch_fields,
                'dtm': dtm_ctrl,
                'citf': citf_ctrl
            })
        else:
            matches += 1

    # Print results
    total_compared = matches + len(mismatches)
    match_rate = (matches / total_compared * 100) if total_compared > 0 else 0

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total frames compared: {total_compared}")
    print(f"  [OK] Matches:    {matches:6d} ({match_rate:.2f}%)")
    print(f"  [!!] Mismatches: {len(mismatches):6d} ({100-match_rate:.2f}%)")

    if len(mismatches) > 0:
        print(f"\nMismatch breakdown:")
        print(f"  Buttons:   {stats['button_mismatches']:6d}")
        print(f"  Stick:     {stats['stick_mismatches']:6d}")
        print(f"  C-Stick:   {stats['substick_mismatches']:6d}")
        print(f"  Triggers:  {stats['trigger_mismatches']:6d}")

    if mismatches and show_mismatches > 0:
        print(f"\n{'='*80}")
        print(f"FIRST {min(show_mismatches, len(mismatches))} MISMATCHES")
        print(f"{'='*80}")

        for i, mismatch in enumerate(mismatches[:show_mismatches]):
            print(f"\nCITF Frame {mismatch['citf_frame']} (DTM Frame {mismatch['dtm_frame']}) - Mismatched: {', '.join(mismatch['fields'])}")
            print(f"  DTM:  buttons=0x{mismatch['dtm']['buttons']:04X}  " +
                  f"stick=({mismatch['dtm']['stickX']:3d},{mismatch['dtm']['stickY']:3d})  " +
                  f"cstick=({mismatch['dtm']['substickX']:3d},{mismatch['dtm']['substickY']:3d})  " +
                  f"L={mismatch['dtm']['triggerLeft']:3d} R={mismatch['dtm']['triggerRight']:3d}")
            print(f"  CITF: buttons=0x{mismatch['citf']['buttons']:04X}  " +
                  f"stick=({mismatch['citf']['stickX']:3d},{mismatch['citf']['stickY']:3d})  " +
                  f"cstick=({mismatch['citf']['substickX']:3d},{mismatch['citf']['substickY']:3d})  " +
                  f"L={mismatch['citf']['triggerLeft']:3d} R={mismatch['citf']['triggerRight']:3d}  " +
                  f"[connected={mismatch['citf']['isConnected']}]")

            # Show button differences in detail
            if 'buttons' in mismatch['fields']:
                dtm_btns = mismatch['dtm']['buttons']
                citf_btns = mismatch['citf']['buttons']
                diff = dtm_btns ^ citf_btns
                print(f"    Button diff: 0x{diff:04X} (DTM raw: 0x{mismatch['dtm'].get('dtm_raw_buttons', 0):04X})")
    else:
        print("\n" + "="*80)
        print("[SUCCESS] ALL CONTROLLER 1 INPUTS MATCH PERFECTLY!")
        print("="*80)

    return len(mismatches) == 0

if __name__ == '__main__':
    dtm_path = Path(r"C:\Users\Brian\Documents\Dolphin Emulator\Citrus Replays\output.dtm")
    citf_path = Path(r"C:\Users\Brian\Documents\Dolphin Emulator\Citrus Replays\output.citframes")

    if not dtm_path.exists():
        print(f"Error: DTM file not found: {dtm_path}")
        sys.exit(1)

    if not citf_path.exists():
        print(f"Error: CITF file not found: {citf_path}")
        sys.exit(1)

    success = compare_controller1(dtm_path, citf_path, sample_every=1, show_mismatches=20)
    sys.exit(0 if success else 1)
