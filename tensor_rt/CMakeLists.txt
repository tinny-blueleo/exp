cmake_minimum_required(VERSION 3.18)
project(sd-tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
find_package(CUDAToolkit REQUIRED)

# TensorRT — search standard paths
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)

find_library(NVINFER_LIB nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB)
    message(FATAL_ERROR
        "TensorRT not found. Install with: sudo apt install tensorrt-dev tensorrt-libs")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "nvinfer: ${NVINFER_LIB}")
message(STATUS "nvonnxparsers: ${NVONNXPARSER_LIB}")

add_executable(sd-tensorrt
    src/main.cpp
    src/engine.cpp
    src/tokenizer.cpp
    src/scheduler.cpp
    src/pipeline.cpp
)

target_include_directories(sd-tensorrt PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(sd-tensorrt PRIVATE
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    CUDA::cudart
)
