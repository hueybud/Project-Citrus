cmake_minimum_required(VERSION 3.13)
project(onnxruntime)

# ONNX Runtime INTERFACE library target.
# Binaries must be placed in:
#   Externals/onnxruntime/include/          <- headers (onnxruntime_cxx_api.h, etc.)
#   Externals/onnxruntime/win-x64/lib/      <- onnxruntime.lib (Windows import lib)
#   Externals/onnxruntime/win-x64/bin/      <- onnxruntime.dll (Windows runtime)
#   Externals/onnxruntime/linux-x64/lib/    <- libonnxruntime.so (Linux)
#
# Download prebuilt releases from:
#   https://github.com/microsoft/onnxruntime/releases
#   Windows: onnxruntime-win-x64-{version}.zip
#   Linux:   onnxruntime-linux-x64-{version}.tgz

add_library(onnxruntime_iface INTERFACE)
add_library(onnxruntime::onnxruntime ALIAS onnxruntime_iface)

target_include_directories(onnxruntime_iface INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(WIN32)
  set(_ORT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/win-x64/lib/onnxruntime.lib")
  set(_ORT_DLL "${CMAKE_CURRENT_SOURCE_DIR}/win-x64/bin/onnxruntime.dll")

  if(NOT EXISTS "${_ORT_LIB}")
    message(FATAL_ERROR
      "ONNX Runtime import library not found: ${_ORT_LIB}\n"
      "Download onnxruntime-win-x64-{version}.zip from "
      "https://github.com/microsoft/onnxruntime/releases and place contents under "
      "Externals/onnxruntime/win-x64/")
  endif()

  target_link_libraries(onnxruntime_iface INTERFACE "${_ORT_LIB}")

  # Copy DLL next to the built executable at link time
  if(EXISTS "${_ORT_DLL}")
    add_custom_target(onnxruntime_dll ALL
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_ORT_DLL}"
        "$<TARGET_FILE_DIR:dolphin-emu>/onnxruntime.dll"
      COMMENT "Copying onnxruntime.dll to output dir")
  endif()

else()
  set(_ORT_SO "${CMAKE_CURRENT_SOURCE_DIR}/linux-x64/lib/libonnxruntime.so")

  if(NOT EXISTS "${_ORT_SO}")
    message(FATAL_ERROR
      "ONNX Runtime shared library not found: ${_ORT_SO}\n"
      "Download onnxruntime-linux-x64-{version}.tgz from "
      "https://github.com/microsoft/onnxruntime/releases and place contents under "
      "Externals/onnxruntime/linux-x64/")
  endif()

  target_link_libraries(onnxruntime_iface INTERFACE "${_ORT_SO}")
endif()
