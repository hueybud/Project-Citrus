// Copyright 2026 Citrus Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "InputCommon/GCPadStatus.h"

namespace Movie
{
// ---------------------------------------------------------------------------
// Model dimensions — must stay in sync with train_transformer.py / build_dataset.py.
// Shared by AIController (feature extraction) and any inference backend.
// ---------------------------------------------------------------------------
namespace AIModelDims
{
constexpr int FEATURE_DIM      = 194;  // total features per frame the model sees
constexpr int BUTTON_DIM_OUT   = 7;    // A, B, X, Y, lob_pass, chip_shot, R
constexpr int STICK_DIM_OUT    = 4;    // stick_x, stick_y, cstick_x, cstick_y
constexpr int PREV_ACTION_DIM  = BUTTON_DIM_OUT + STICK_DIM_OUT;        // 11
constexpr int CORE_FEATURE_DIM = FEATURE_DIM - PREV_ACTION_DIM;         // 183
// Transformer KV cache: [layers, 2 (k/v), 1 (batch), seq_len-1, dim]
constexpr int KV_CACHE_LAYERS  = 3;
constexpr int KV_CACHE_SEQ     = 127;  // SEQ_LEN(128) - 1
constexpr int KV_CACHE_DIM     = 512;
constexpr int KV_CACHE_SIZE    = KV_CACHE_LAYERS * 2 * KV_CACHE_SEQ * KV_CACHE_DIM;
}  // namespace AIModelDims

// ---------------------------------------------------------------------------
// AIInputFrame — emu-thread → backend handoff packet.
// Contains the 183-float CORE features (built by AIController on the emu
// thread from live GC memory), plus flags the backend needs to decide
// behavior.  Cheap to construct; expensive work happens on the backend side.
// ---------------------------------------------------------------------------
struct AIInputFrame
{
  std::vector<float> core_features;          // CORE_FEATURE_DIM floats
  bool               reset_context = false;  // backend zeros KV + prev_labels first
  bool               mirror_x      = false;  // stick_x sign flip in DecodeOutput

  // Reward / episode-bookkeeping scalars used by the IpcBackend (Python RL
  // trainer needs these alongside the feature vector to compute rewards and
  // detect terminal events).  LocalOnnxBackend ignores them.
  uint32_t frame_id    = 0;     // monotonically incremented per submitted frame
  uint16_t score_left  = 0;
  uint16_t score_right = 0;
};

// ---------------------------------------------------------------------------
// AIInferenceBackend — pluggable inference site.
//
// LocalOnnxBackend     — runs ONNX Runtime on a dedicated worker thread.
// IpcBackend (future)  — round-trips AIInputFrame to a Python RL trainer
//                        over a TCP socket; same Submit/GetLastOutput API.
//
// Threading contract:
//   Submit()         — called from the emu thread; non-blocking; latest-wins
//                      (an unread previous frame is overwritten).
//   GetLastOutput()  — safe from any thread.
//   HasOutput()      — safe from any thread.
//   Shutdown()       — emu thread; blocks until worker is joined.
// ---------------------------------------------------------------------------
class AIInferenceBackend
{
public:
  virtual ~AIInferenceBackend() = default;

  virtual void        Submit(AIInputFrame frame)      = 0;
  virtual GCPadStatus GetLastOutput() const           = 0;
  virtual bool        HasOutput() const               = 0;
  virtual void        Shutdown()                      = 0;
};

// ---------------------------------------------------------------------------
// AIController — emu-thread façade.  Reads game state, builds the input
// frame, hands off to a backend.  Owns nothing model-specific (no ORT
// session, no KV cache, no prev_labels) — that all lives in the backend.
// ---------------------------------------------------------------------------
class AIController
{
public:
  // Reset callback signature for the IPC backend: invoked from a worker
  // thread when the Python trainer sends a control message asking Dolphin to
  // reload a savestate.  Implementations should marshal back to the host /
  // emu thread before touching Core::State.
  using ResetCallback = std::function<void(uint32_t savestate_id)>;

  AIController();
  ~AIController();

  // Load and start the local ONNX backend.  Returns false on any failure.
  bool Load(const std::string& onnx_path);

  // Load and start the IPC backend.  Listens on TCP loopback `port` for a
  // single Python client.  reset_cb is invoked from the IPC receiver thread
  // when a reset control message arrives.  Returns false if bind/listen fails.
  bool LoadIpc(int port, ResetCallback reset_cb);

  void Shutdown();
  bool IsLoaded() const;

  // Called once per rendered frame from TickAIController() in Movie.cpp.
  // Reads game state on the emu thread, submits to the backend asynchronously.
  // controlled_port: GC port index (0-3) the AI controls.
  // mirror_x:        true when the AI's team attacks left — X coords are
  //                  negated so the model always sees "attacks right".
  void OnFrameEnd(int controlled_port, bool mirror_x);

  // True when we're inside an active-play phase AND the backend has an
  // output to deliver.  PlayController() guards on this before injecting.
  bool IsMatchActive() const;

  // True when the game is in the goal-replay/transition phase.
  // Static so Movie.cpp can call it without a full inference session running.
  static bool IsGoalReplay();

  // Returns the raw eGameState value for logging.
  static uint32_t GetGamePhase();

  // Delivered to PlayController() for the controlled port.  Stale by at
  // most one frame under normal operation (backend is keeping up).
  GCPadStatus GetLastOutput() const;

private:
  // Build the 183-float CORE feature vector from live GC memory.
  // MUST be called on the emu thread (reads GC memory).
  std::vector<float> ReadGameStateCore(int controlled_port, bool mirror_x) const;

  std::unique_ptr<AIInferenceBackend> m_backend;

  // Phase tracking — only the emu thread can read GC memory.  Controller
  // owns this; the backend only sees a reset_context boolean per frame.
  int  m_prev_phase_family = -1;
  bool m_phase_active      = false;

  // Monotonic per-controller frame id; echoed by the IPC client for
  // stale-frame detection.  Wraps at 2^32 (~828 days @ 60Hz, fine).
  uint32_t m_next_frame_id = 1;
};

}  // namespace Movie
