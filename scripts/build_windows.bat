@echo off
REM BikeGuard Windows Build Script
REM This script builds the BikeGuard C++ native bridge for Windows

echo ========================================
echo BikeGuard Windows Build Script
echo ========================================

REM Check if vcpkg is installed
if not defined VCPKG_ROOT (
    echo ERROR: VCPKG_ROOT environment variable is not set
    echo Please install vcpkg and set VCPKG_ROOT environment variable
    echo Installation guide: https://vcpkg.io/en/getting-started.html
    pause
    exit /b 1
)

echo Using vcpkg at: %VCPKG_ROOT%

REM Create build directory
if not exist build (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" ^
         -DCMAKE_BUILD_TYPE=Release ^
         -DVCPKG_TARGET_TRIPLET=x64-windows

if %ERRORLEVEL% neq 0 (
    echo ERROR: CMake configuration failed
    pause
    exit /b 1
)

REM Build the project
echo Building project...
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo ========================================
echo Build completed successfully!
echo ========================================
echo Executable location: build\Release\BikeGuard.exe
echo.
echo To run the application:
echo   build\Release\BikeGuard.exe
echo.
echo Note: Make sure you have a compatible camera connected
echo and place your ONNX model in the models/ directory
pause
