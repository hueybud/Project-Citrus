# ONNX Runtime — Prebuilt Binaries

Place prebuilt ONNX Runtime files here before building.

## Download

From https://github.com/microsoft/onnxruntime/releases, download:
- **Windows x64**: `onnxruntime-win-x64-{version}.zip`
- **Linux x64**: `onnxruntime-linux-x64-{version}.tgz`

Recommended version: 1.21.0 (or later stable)

## Layout after extraction

```
Externals/onnxruntime/
├── include/
│   ├── onnxruntime_c_api.h
│   ├── onnxruntime_cxx_api.h
│   ├── onnxruntime_cxx_inline.h
│   └── ...
├── win-x64/
│   ├── lib/
│   │   └── onnxruntime.lib        ← import library (from zip's lib/ folder)
│   └── bin/
│       └── onnxruntime.dll        ← runtime DLL (from zip's lib/ folder)
└── linux-x64/
    └── lib/
        └── libonnxruntime.so      ← shared library (from tgz's lib/ folder)
```

## Windows quick setup (PowerShell)

```powershell
$ver = "1.21.0"
$dir = "Externals/onnxruntime"
Invoke-WebRequest "https://github.com/microsoft/onnxruntime/releases/download/v$ver/onnxruntime-win-x64-$ver.zip" -OutFile "ort.zip"
Expand-Archive ort.zip -DestinationPath ort_tmp
Copy-Item ort_tmp/onnxruntime-win-x64-$ver/include/* $dir/include/
Copy-Item ort_tmp/onnxruntime-win-x64-$ver/lib/onnxruntime.lib $dir/win-x64/lib/
Copy-Item ort_tmp/onnxruntime-win-x64-$ver/lib/onnxruntime.dll $dir/win-x64/bin/
Remove-Item ort.zip, ort_tmp -Recurse
```

## Linux quick setup (bash)

```bash
VER=1.21.0
DIR=Externals/onnxruntime
wget "https://github.com/microsoft/onnxruntime/releases/download/v$VER/onnxruntime-linux-x64-$VER.tgz"
tar xf onnxruntime-linux-x64-$VER.tgz
cp onnxruntime-linux-x64-$VER/include/* $DIR/include/
cp onnxruntime-linux-x64-$VER/lib/libonnxruntime.so* $DIR/linux-x64/lib/
rm -rf onnxruntime-linux-x64-$VER onnxruntime-linux-x64-$VER.tgz
```
