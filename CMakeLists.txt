cmake_minimum_required(VERSION 3.20)
project(BikeGuard VERSION 1.0.0 LANGUAGES CXX)

# C++20 standard with Windows optimizations
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Windows-specific configuration
if(WIN32)
    set(CMAKE_SYSTEM_VERSION 10.0)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX _WIN32_WINNT=0x0A00)
endif()

# MSVC specific optimizations for high-performance Windows applications
if(MSVC)
    # Enable all warnings and treat them as errors
    add_compile_options(/W4 /WX /utf-8 /permissive-)
    
    # Windows-specific definitions
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
    
    # High-performance optimizations for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Ob2 /Ot /GL /DNDEBUG)
        add_link_options(/LTCG /OPT:REF /OPT:ICF)
        
        # Enable parallel compilation
        add_compile_options(/MP)
        
        # Favor fast code and small code
        add_compile_options(/fp:fast)
    endif()
    
    # Debug configuration
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi /RTC1)
        add_link_options(/DEBUG)
    endif()
    
    # Enable C++20 features and conformance
    add_compile_options(/std:c++20 /Zc:__cplusplus)
    
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG -march=native)
    endif()
endif()

# vcpkg integration with automatic detection
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# Find packages - OpenCV World and ONNX Runtime with DirectML
find_package(OpenCV REQUIRED COMPONENTS world)
find_package(onnxruntime CONFIG REQUIRED)

# Windows Media Foundation for camera capture
if(WIN32)
    find_package(mf REQUIRED)
    find_package(mfplat REQUIRED)
    find_package(mfuuid REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files - modern C++20 structure
file(GLOB_RECURSE CORE_SOURCES "src/core/*.cpp" "src/core/*.hpp")
file(GLOB_RECURSE PLATFORM_SOURCES "src/platform/*.cpp" "src/platform/*.hpp")
file(GLOB_RECURSE UTILS_SOURCES "src/utils/*.cpp" "src/utils/*.hpp")
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")

set(ALL_SOURCES ${CORE_SOURCES} ${PLATFORM_SOURCES} ${UTILS_SOURCES})

# Create static library for core engine
add_library(bikeguard_engine STATIC ${ALL_SOURCES} ${HEADERS})

# Target configuration for bikeguard_engine
target_include_directories(bikeguard_engine PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries for core engine
target_link_libraries(bikeguard_engine 
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
)

# Windows-specific libraries for Media Foundation and system APIs
if(WIN32)
    target_link_libraries(bikeguard_engine
        # Media Foundation libraries
        mf
        mfplat
        mfuuid
        
        # Core Windows libraries
        kernel32
        user32
        gdi32
        ole32
        oleaut32
        uuid
        advapi32
        setupapi
        
        # DirectX libraries for hardware acceleration
        dxgi
        d3d11
        d3d12
        
        # Windows Runtime
        runtimeobject
    )
    
    # Windows-specific compile definitions
    target_compile_definitions(bikeguard_engine PRIVATE 
        _WIN32_WINNT=0x0A00  # Windows 10
        WIN32_LEAN_AND_MEAN   # Reduce Windows header size
        NOMINMAX              # Prevent min/max macro conflicts
    )
endif()

# Create main executable
add_executable(BikeGuard src/main.cpp)
target_link_libraries(BikeGuard bikeguard_engine)

# Windows executable configuration
if(WIN32)
    # Set subsystem to Windows for GUI applications
    set_target_properties(BikeGuard PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
    
    # Embed manifest for Windows 10/11 compatibility
    set(MANIFEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/app.manifest")
    if(EXISTS ${MANIFEST_FILE})
        target_sources(BikeGuard PRIVATE ${MANIFEST_FILE})
    endif()
endif()

# Installation configuration
install(TARGETS BikeGuard bikeguard_engine
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# Copy models and resources to build directory
file(GLOB MODEL_FILES "models/*")
if(MODEL_FILES)
    file(COPY ${MODEL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/models)
endif()

# Copy any configuration files
file(GLOB CONFIG_FILES "config/*")
if(CONFIG_FILES)
    file(COPY ${CONFIG_FILES} DESTINATION ${CMAKE_BINARY_DIR}/config)
endif()

# Performance profiling configuration
option(BIKEGUARD_ENABLE_PROFILING "Enable performance profiling" OFF)
if(BIKEGUARD_ENABLE_PROFILING)
    target_compile_definitions(bikeguard_engine PRIVATE BIKEGUARD_PROFILING)
endif()

# Development utilities
option(BIKEGUARD_BUILD_TESTS "Build test suite" OFF)
if(BIKEGUARD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation generation
option(BIKEGUARD_BUILD_DOCS "Build documentation" OFF)
if(BIKEGUARD_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    endif()
endif()