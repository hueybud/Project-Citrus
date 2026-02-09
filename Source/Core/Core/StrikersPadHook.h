#pragma once

#include "Common/CommonTypes.h"

// Interface for hooking into Super Mario Strikers' PAD reading function
// This provides a two-way door: read inputs for CITF capture, write inputs for AI inference

namespace StrikersPadHook
{

struct PadData
{
  u16 buttons;       // Button bitmask
  u8 stickX;         // Main stick X (0-255, centered at 128)
  u8 stickY;         // Main stick Y (0-255, centered at 128)
  u8 cStickX;        // C-stick X (0-255, centered at 128)
  u8 cStickY;        // C-stick Y (0-255, centered at 128)
  u8 triggerL;       // L trigger (0-255)
  u8 triggerR;       // R trigger (0-255)
  u8 analogA;        // Analog A (0-255)
  u8 analogB;        // Analog B (0-255)
  u8 error;          // Error/connected status (0 = connected)
};

// Get the last captured pad data for a controller (0-3)
PadData GetPadData(int controller);

// Set whether AI mode is active (for future AI inference)
void SetAIMode(bool enabled);

// Check if AI mode is active
bool IsAIModeActive();

// HLE hook function - called by HLE system at 0x801c3808
void HookFunction();

// Initialize the hook system
void Initialize();

// Shutdown the hook system
void Shutdown();

}  // namespace StrikersPadHook
