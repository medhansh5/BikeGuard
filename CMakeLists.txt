cmake_minimum_required(VERSION 3.10)
project(BikeGuard)

# Use C++17 for better performance and filesystem features
set(CMAKE_CXX_STANDARD 17)

# 1. Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 2. Find ONNX Runtime (You'll need to set the path to where you downloaded it)
# set(ONNXRUNTIME_ROOT "/path/to/onnxruntime") 
# include_directories(${ONNXRUNTIME_ROOT}/include)

# 3. Create the executable
add_executable(BikeGuard src/main.cpp)

# 4. Link the libraries
target_link_libraries(BikeGuard ${OpenCV_LIBS})
# target_link_libraries(BikeGuard ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)