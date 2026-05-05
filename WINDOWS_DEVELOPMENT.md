# BikeGuard Windows Development Guide

This guide covers setting up and building the BikeGuard C++ native bridge on Windows with MSVC and vcpkg.

## Prerequisites

### System Requirements
- Windows 10/11 (x64)
- Visual Studio 2019 or 2022 with C++ development tools
- Git for Windows
- CMake 3.16 or higher

### Required Software
1. **Visual Studio** - Install with "Desktop development with C++" workload
2. **CMake** - Download from https://cmake.org/download/
3. **Git** - Download from https://git-scm.com/download/win

## Setup Instructions

### 1. Install vcpkg

Run the setup script to automatically install vcpkg and dependencies:

```batch
scripts\setup_vcpkg.bat
```

Or manually install vcpkg:

```batch
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

### 2. Set Environment Variables

Add vcpkg to your system environment:

```batch
set VCPKG_ROOT=C:\path\to\vcpkg
set PATH=%VCPKG_ROOT%;%PATH%
```

For permanent setup, add these to Windows System Environment Variables.

### 3. Install Dependencies

```batch
vcpkg install opencv[contrib,dnn,highgui]:x64-windows
vcpkg install onnxruntime[dml]:x64-windows
```

## Building the Project

### Quick Build

Use the provided build script:

```batch
scripts\build_windows.bat
```

### Manual Build

```batch
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
         -DCMAKE_BUILD_TYPE=Release ^
         -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build . --config Release
```

## Project Structure

```
Bikeguard/
├── include/
│   └── bikeguard_core.hpp          # Main API header
├── src/
│   ├── inference_engine.cpp        # Core inference logic
│   ├── hardware_abstraction.cpp    # Platform-specific implementations
│   └── main.cpp                    # Demo application
├── scripts/
│   ├── build_windows.bat          # Build script
│   └── setup_vcpkg.bat            # vcpkg setup
├── models/                         # Model files directory
├── vcpkg.json                     # vcpkg dependency manifest
├── CMakeLists.txt                 # CMake configuration
└── WINDOWS_DEVELOPMENT.md         # This file
```

## Architecture Overview

### Core Components

1. **InferenceEngine**: Zero-allocation inference loop with ONNX Runtime
2. **CameraCapture**: Abstract camera interface with OpenCV/DirectX implementations
3. **HardwareAccelerator**: Hardware abstraction for CPU/DirectML acceleration

### Platform-Specific Features

#### Windows-Specific Optimizations
- DirectML GPU acceleration via ONNX Runtime
- DirectX camera capture (planned)
- Windows-specific memory management
- MSVC compiler optimizations

#### Cross-Platform Compatibility
- `#ifdef _WIN32` blocks for Windows-specific code
- Portable core logic for Linux migration
- Abstract interfaces for hardware abstraction

## Usage Example

```cpp
#include "bikeguard_core.hpp"

int main() {
    try {
        // Initialize configuration
        BikeGuard::InferenceConfig config;
        config.model_path = "models/helmet_detector.onnx";
        config.use_gpu = true;
        config.confidence_threshold = 0.5f;
        
        // Create inference engine
        auto engine = std::make_unique<BikeGuard::InferenceEngine>();
        engine->initialize(config);
        
        // Create camera capture
        auto camera = BikeGuard::create_opencv_camera();
        camera->initialize(0);
        
        cv::Mat frame;
        while (camera->capture_frame(frame)) {
            // Run inference
            auto detections = engine->run_inference(frame);
            
            // Draw results
            BikeGuard::utils::draw_detections(frame, detections);
            
            // Display
            cv::imshow("BikeGuard", frame);
            if (cv::waitKey(1) == 'q') break;
            
            // Print performance metrics
            auto metrics = engine->get_metrics();
            std::cout << "FPS: " << metrics.fps << std::endl;
        }
        
    } catch (const BikeGuard::BikeGuardException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
```

## Performance Optimizations

### Zero-Allocation Design
- Pre-allocated buffers for inference
- Memory reuse between frames
- Minimal dynamic allocations in hot path

### Hardware Acceleration
- DirectML GPU acceleration on Windows
- SIMD optimizations in OpenCV
- Multi-threading support in ONNX Runtime

### Compiler Optimizations
- MSVC /O2 optimizations for Release builds
- Link-time optimization (LTO)
- Profile-guided optimization (PGO) ready

## Troubleshooting

### Common Issues

1. **vcpkg not found**
   - Ensure VCPKG_ROOT environment variable is set
   - Verify vcpkg installation path

2. **Build fails with MSVC errors**
   - Ensure Visual Studio C++ tools are installed
   - Check CMake generator matches your VS version

3. **Runtime DLL errors**
   - Ensure vcpkg installed binaries are in PATH
   - Check that correct architecture (x64) is used

4. **GPU acceleration not working**
   - Verify DirectML-compatible GPU is available
   - Check that onnxruntime[dml] package is installed

### Debug Build

For debugging, build with Debug configuration:

```batch
cmake --build . --config Debug
```

### Clean Build

To clean and rebuild:

```batch
rmdir /s /q build
scripts\build_windows.bat
```

## Model Requirements

The system expects ONNX models in YOLOv8 format:
- Input: `[1, 3, 640, 640]` (batch, channels, height, width)
- Output: `[1, N, 85]` where N = number of detections, 85 = 4(bbox) + 1(conf) + 80(classes)

Place your trained model in the `models/` directory:
```
models/
└── helmet_detector.onnx
```

## Migration to Linux

The codebase is designed for easy Linux migration:
- Core logic is platform-agnostic
- Windows-specific code is isolated in `#ifdef _WIN32` blocks
- CMake configuration supports Linux toolchains
- vcpkg works on Linux for dependency management

To migrate:
1. Replace Windows-specific implementations with Linux equivalents
2. Update CMakeLists.txt for Linux toolchain
3. Use vcpkg Linux packages or system package managers

## Performance Benchmarks

Expected performance on typical hardware:
- **CPU (Intel i7)**: 15-25 FPS
- **GPU (DirectML)**: 30-60+ FPS
- **Memory usage**: <500MB (including model)
- **Startup time**: <2 seconds

## Contributing

When contributing to the Windows build:
1. Test with both Debug and Release configurations
2. Verify MSVC compatibility
3. Test on different Windows versions
4. Ensure vcpkg dependencies are correctly specified
5. Update this documentation for any API changes
