cmake_minimum_required(VERSION 3.18)
project(cuda_fun_project LANGUAGES CXX CUDA)

# Set CUDA standards
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Specify CUDA architecture for Quadro P620 (Compute Capability 6.1)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Executable
add_executable(cuda_fun_project src/main.cu)

# Link CUDA libraries
target_link_libraries(cuda_fun_project ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})

# CUDA compiler flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

# Print CUDA paths for debugging
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
