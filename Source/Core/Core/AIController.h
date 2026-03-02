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
class SessionOptions;
class MemoryInfo;
}  // namespace Ort

#include "InputCommon/GCPadStatus.h"

namespace Movie
{
// AI controller backed by an ONNX Runtime model.
// One instance is created when a model path is configured in Dolphin.ini.
// OnFrameEnd() reads live GC memory, runs inference, and caches the result.
// GetLastOutput() is called from PlayController() (up to 2× per game frame).
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

  // Called once per rendered frame from OnFrameEnd() in Core.cpp.
  // Reads current GC game state, runs inference, stores result in m_last_output.
  // controlled_port: GC port index (0-3) that the AI controls.
  // mirror_x:        true when the AI's team attacks left (RIGHT team) — X coords are
  //                  negated so the model always sees the canonical "attacks right" view.
  void OnFrameEnd(int controlled_port, bool mirror_x);

  // True only when the last OnFrameEnd() call successfully ran inference
  // (i.e. gamePhase was active play 4/5). False during menus / goal celebrations.
  bool IsMatchActive() const;

  // Called from PlayController() for the controlled port.
  // Returns the output cached by the most recent OnFrameEnd() call.
  GCPadStatus GetLastOutput() const;

private:
  static constexpr int WINDOW_SIZE = 8;
  static constexpr int FEATURE_DIM = 430;                    // features per single frame
  static constexpr int INPUT_DIM   = WINDOW_SIZE * FEATURE_DIM;  // 3440 — ORT input width

  // Build the 430-float feature vector from live GC memory.
  // Mirrors the Python extract_features() in Tools/build_dataset.py exactly.
  std::vector<float> ReadGameState(int controlled_port, bool mirror_x) const;

  // Decode raw model outputs into a GCPadStatus.
  // btn_probs[6]: sigmoid probabilities for A, B, X, Y, L, R
  // stick_vals[4]: tanh-activated values for stick_x, stick_y, cstick_x, cstick_y
  // mirror_x: if true, negate stick_x before converting to byte (undo canonical flip)
  GCPadStatus DecodeOutput(const float* btn_probs, const float* stick_vals,
                           bool mirror_x) const;

  std::unique_ptr<Ort::Env>     m_env;
  std::unique_ptr<Ort::Session> m_session;
  GCPadStatus                   m_last_output{};
  bool                          m_loaded       = false;
  bool                          m_match_active = false;

  // Circular frame buffer for temporal stacking (WINDOW_SIZE=8 frames).
  // m_buffer_head: index of the next slot to write.
  // m_buffer_count: 0 until first active frame, then WINDOW_SIZE thereafter.
  // On the first active frame after a reset, all slots are filled with that frame
  // (warmup replication) so inference can run immediately without zero-padding.
  std::array<std::array<float, FEATURE_DIM>, WINDOW_SIZE> m_frame_buffer{};
  int m_buffer_head  = 0;
  int m_buffer_count = 0;
};

}  // namespace Movie
