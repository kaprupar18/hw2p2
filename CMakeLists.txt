cmake_minimum_required (VERSION 2.8.0)
project (hw2p2)

configure_file(auto-bridges-gpu auto-bridges-gpu COPYONLY)
configure_file(job-bridges-gpu job-bridges-gpu COPYONLY)
configure_file(job-bridges-serial job-bridges-serial COPYONLY)

find_package(MPI)
find_package(CUDA)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(CMAKE_NVCC_COMPILER nvcc)

set(CUDA_NVCC_FLAGS -O3 -gencode arch=compute_37,code=sm_37)

set(HEADERS
    common.h
    
)

set(SOURCES
    #common.cpp
)

CUDA_ADD_EXECUTABLE(matrix ${HEADERS} matrix.cu common.cu ${CUDA_LIBRARIES})
target_link_libraries( matrix)

CUDA_ADD_EXECUTABLE(serial ${HEADERS} serial.cu common.cu ${CUDA_LIBRARIES})
target_link_libraries( serial)

CUDA_ADD_EXECUTABLE(autograder ${HEADERS} autograder.cu common.cu ${CUDA_LIBRARIES})
target_link_libraries( autograder)

CUDA_ADD_EXECUTABLE(gpu ${HEADERS} gpu.cu common.cu ${CUDA_LIBRARIES})
target_link_libraries( gpu)

