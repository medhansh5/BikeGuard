@echo off
REM vcpkg Setup Script for BikeGuard
REM This script installs and configures vcpkg for Windows development

echo ========================================
echo BikeGuard vcpkg Setup Script
echo ========================================

REM Check if vcpkg is already installed
if exist "%VCPKG_ROOT%\vcpkg.exe" (
    echo vcpkg already installed at: %VCPKG_ROOT%
    goto :install_packages
)

echo Installing vcpkg...

REM Clone vcpkg repository
if not exist vcpkg (
    git clone https://github.com/Microsoft/vcpkg.git
)

cd vcpkg

REM Bootstrap vcpkg
call bootstrap-vcpkg.bat

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to bootstrap vcpkg
    pause
    exit /b 1
)

REM Set VCPKG_ROOT environment variable for current session
set VCPKG_ROOT=%CD%
echo VCPKG_ROOT set to: %VCPKG_ROOT%

REM Add vcpkg to PATH for current session
set PATH=%VCPKG_ROOT%;%PATH%

cd ..

:install_packages
echo Installing required packages...

REM Install OpenCV with contrib modules
echo Installing OpenCV...
"%VCPKG_ROOT%\vcpkg.exe" install opencv[contrib,dnn,highgui]:x64-windows

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install OpenCV
    pause
    exit /b 1
)

REM Install ONNX Runtime with DirectML support
echo Installing ONNX Runtime...
"%VCPKG_ROOT%\vcpkg.exe" install onnxruntime[dml]:x64-windows

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install ONNX Runtime
    pause
    exit /b 1
)

echo ========================================
echo vcpkg setup completed successfully!
echo ========================================
echo.
echo To make vcpkg permanently available, set the VCPKG_ROOT
echo environment variable to: %VCPKG_ROOT%
echo.
echo You can now run the build script:
echo   scripts\build_windows.bat
pause
