# ─────────────────────────────────────────────────────────────────────────────
# Citrus Dolphin — CIT→CITF batch conversion image
#
# Prerequisites (run on host before docker build):
#   git submodule update --init --recursive
#
# Build:
#   docker build -t citrus-converter .
#
# Run:
#   docker run --rm \
#     -v /path/to/cits:/data/cits \
#     -v /path/to/game.iso:/data/game.iso:ro \
#     citrus-converter /data/cits
#
# Optional env overrides:
#   SMS_ISO      — path inside the container to the SMS ISO (default: /data/game.iso)
#   DOLPHIN_EXE  — path to the dolphin-emu-nogui binary   (default: /opt/dolphin/dolphin-emu-nogui)
#   DOLPHIN_INI  — path to Dolphin.ini                    (default: /home/dolphin/.config/dolphin-emu/Dolphin.ini)
# ─────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: build ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    nasm \
    wget \
    ca-certificates \
    libevdev-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    libssl-dev \
    libasound2-dev \
    libpulse-dev \
    libcurl4-openssl-dev \
    libsfml-dev \
    libreadline-dev \
    zlib1g-dev \
    libpng-dev \
    libvulkan-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build/dolphin

# Copy the full source tree (submodules must be initialized on the host first).
COPY . .

# Download ONNX Runtime Linux binaries + headers.
# This overwrites any Windows-only headers already in the build context with the
# same platform-neutral headers from the Linux tarball — safe and idempotent.
ARG ORT_VERSION=1.20.1
RUN curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
      -o ort.tgz \
 && tar xf ort.tgz \
 && mkdir -p Externals/onnxruntime/include Externals/onnxruntime/linux-x64/lib \
 && cp onnxruntime-linux-x64-${ORT_VERSION}/include/* Externals/onnxruntime/include/ \
 && cp onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so* Externals/onnxruntime/linux-x64/lib/ \
 && rm -rf onnxruntime-linux-x64-${ORT_VERSION} ort.tgz

# Configure and build DolphinNoGUI (headless, no Qt, no X11).
RUN cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DLINUX_LOCAL_DEV=true \
      -DENABLE_QT=OFF \
      -DENABLE_X11=OFF \
 && cmake --build build --parallel $(nproc)


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime libraries that dolphin-emu-nogui needs.
# Ubuntu 24.04 renamed several packages with the t64 suffix (libc ABI transition).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libevdev2 \
    libusb-1.0-0 \
    libudev1 \
    libssl3 \
    libasound2t64 \
    libpulse0 \
    libcurl4t64 \
    libsfml-network2.6 \
    libreadline8t64 \
    zlib1g \
    libpng16-16 \
    libvulkan1 \
    python3 \
    python3-zstandard \
  && rm -rf /var/lib/apt/lists/*

# ── Dolphin binary + game data ────────────────────────────────────────────────
# Binary is placed in /opt/dolphin/.  With LINUX_LOCAL_DEV, Dolphin looks for
# the Sys directory in the same directory as the executable.
RUN mkdir -p /opt/dolphin
COPY --from=builder /build/dolphin/build/Binaries/dolphin-emu-nogui /opt/dolphin/
COPY --from=builder /build/dolphin/Data/Sys                          /opt/dolphin/Sys/

# ── ONNX Runtime shared library ───────────────────────────────────────────────
COPY --from=builder /build/dolphin/Externals/onnxruntime/linux-x64/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

# ── Conversion script ─────────────────────────────────────────────────────────
COPY Tools/convert_cits.py /usr/local/bin/convert_cits.py
RUN chmod +x /usr/local/bin/convert_cits.py

# ── Dolphin config ────────────────────────────────────────────────────────────
# Pre-create the config directory and a minimal Dolphin.ini with UseNullBackend
# enabled so every container run uses the null graphics backend by default.
# The convert_cits.py script will overwrite this at startup — this just ensures
# the directory exists before Dolphin first launches.
ENV HOME=/home/dolphin
RUN mkdir -p /home/dolphin/.config/dolphin-emu \
 && printf '[Movie]\nUseNullBackend = True\n' \
      > /home/dolphin/.config/dolphin-emu/Dolphin.ini

# ── Runtime defaults ──────────────────────────────────────────────────────────
# Override with -e SMS_ISO=... / -e DOLPHIN_EXE=... if your layout differs.
ENV DOLPHIN_EXE=/opt/dolphin/dolphin-emu-nogui
ENV SMS_ISO=/data/game.iso

WORKDIR /opt/dolphin

ENTRYPOINT ["python3", "/usr/local/bin/convert_cits.py"]
CMD ["--help"]
