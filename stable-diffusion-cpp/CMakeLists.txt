cmake_minimum_required(VERSION 3.12)
project("sd-experiment" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Build stable-diffusion.cpp as a subdirectory (it provides the "stable-diffusion" target)
add_subdirectory(stable-diffusion.cpp)

add_executable(sd-experiment main.cpp)
target_link_libraries(sd-experiment PRIVATE stable-diffusion)
target_compile_features(sd-experiment PRIVATE cxx_std_17)

# stb headers live in the submodule's thirdparty dir
target_include_directories(sd-experiment PRIVATE stable-diffusion.cpp/thirdparty)
