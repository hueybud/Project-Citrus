// Copyright 2026 Citrus Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

// Forward-declare ONNX Runtime types to keep this header light.
namespace Ort
{
class Env;
class Session;
}  // namespace Ort

#include "InputCommon/GCPadStatus.h"

namespace Movie
{
// AI controller backed by an ONNX Runtime transformer model.
// One instance is created when a model path is configured in Dolphin.ini.
// OnFrameEnd() reads live GC memory, runs inference, and caches the result.
// GetLastOutput() is called from PlayController() (up to 2x per game frame).
class AIController
{
public:
  AIController();
  ~AIController();

  // Load model from an ONNX file.  Returns false if the file is missing or
  // the model has the wrong input/output shape.
  bool Load(const std::string& onnx_path);
  void Shutdown();
  bool IsLoaded() const;

  // Called once per rendered frame from TickAIController() in Movie.cpp.
  // Reads current GC game state, runs inference, stores result in m_last_output.
  // controlled_port: GC port index (0-3) that the AI controls.
  // mirror_x:        true when the AI's team attacks left (RIGHT team) — X coords
  //                  are negated so the model always sees "attacks right".
  void OnFrameEnd(int controlled_port, bool mirror_x);

  // True only when the last OnFrameEnd() call successfully ran inference.
  bool IsMatchActive() const;

  // Returns the GCPadStatus cached by the most recent OnFrameEnd() call.
  GCPadStatus GetLastOutput() const;

private:
  // ── Model dimensions (must stay in sync with train_transformer.py v6) ───────
  static constexpr int FEATURE_DIM     = 194;  // flat features per frame (v6)
  static constexpr int BUTTON_DIM_OUT  = 7;    // A, B, X, Y, lob_pass, chip_shot, R
  static constexpr int STICK_DIM_OUT   = 4;    // stick_x, stick_y, cstick_x, cstick_y
  static constexpr int PREV_ACTION_DIM = BUTTON_DIM_OUT + STICK_DIM_OUT;  // 11

  // Transformer KV cache: [TEMPORAL_LAYERS, 2 (k/v), 1 (batch), SEQ_LEN-1, TEMPORAL_DIM]
  static constexpr int KV_CACHE_LAYERS = 3;
  static constexpr int KV_CACHE_SEQ    = 127;  // SEQ_LEN(128) - 1
  static constexpr int KV_CACHE_DIM    = 512;
  // Flat size: 3 * 2 * 1 * 127 * 512 = 389,112
  static constexpr int KV_CACHE_SIZE   = KV_CACHE_LAYERS * 2 * KV_CACHE_SEQ * KV_CACHE_DIM;

  // Build the 194-float feature vector from live GC memory.
  // Mirrors extract_features() + prev_labels in build_dataset.py v6 exactly.
  std::vector<float> ReadGameState(int controlled_port, bool mirror_x) const;

  // Decode raw model outputs (btn_probs [7], stick_vals [4]) into a GCPadStatus.
  // btn_probs are 0/1 flags from the action vocabulary categorical output.
  // mirror_x: if true, negate stick_x/cstick_x before converting to byte.
  // lob_pass (index 4) maps to L+A, chip_shot (index 5) maps to L+B.
  GCPadStatus DecodeOutput(const float* btn_probs, const float* stick_vals,
                            bool mirror_x);

  std::unique_ptr<Ort::Env>     m_env;
  std::unique_ptr<Ort::Session> m_session;
  GCPadStatus                   m_last_output{};
  bool                          m_loaded       = false;
  bool                          m_match_active = false;

  // Transformer KV cache — carries temporal context across game frames.
  // Stores [KV_CACHE_LAYERS, 2, 1, KV_CACHE_SEQ, KV_CACHE_DIM] in row-major order.
  // Reset to zeros on phase-family transitions and when match becomes inactive.
  std::array<float, KV_CACHE_SIZE> m_kv_cache{};

  // Previous frame's label vector (A B X Y L R stick_x stick_y cstick_x cstick_y).
  // Appended as the last PREV_ACTION_DIM features each frame.
  // Zeroed on phase transitions to match training segment boundaries.
  std::array<float, PREV_ACTION_DIM> m_prev_labels{};

  // Phase family from the last OnFrameEnd() call: 1=kickoff, 4=active, -1=other.
  // Used to detect segment boundaries and reset context state accordingly.
  int m_prev_phase_family = -1;
};

}  // namespace Movie
