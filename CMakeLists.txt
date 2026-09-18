cmake_minimum_required(VERSION 3.22)
project(HITS)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS_DEBUG "-march=native ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native  ${CMAKE_CXX_FLAGS_RELEASE}")

set(gpu_device torch::kCUDA)

# Local writable data root. Parameter.h.in expects the value to already be quoted.
if(NOT data_father_path)
    set(data_father_path "\"${PROJECT_SOURCE_DIR}/data/\"")
endif()

# Locate libtorch: honour an explicit -DTorch_DIR / $ENV{Torch_DIR}, otherwise use
# the torch that ships with the python installation.
if(NOT Torch_DIR)
    file(GLOB _torch_candidates
            "$ENV{Torch_DIR}"
            "/usr/local/lib/python3*/dist-packages/torch/share/cmake/Torch"
            "${PROJECT_SOURCE_DIR}/libtorch/share/cmake/Torch")
    foreach(_cand ${_torch_candidates})
        if(EXISTS "${_cand}/TorchConfig.cmake")
            set(Torch_DIR "${_cand}")
            break()
        endif()
    endforeach()
endif()
message(STATUS "Torch_DIR = ${Torch_DIR}")
message(STATUS "data_father_path = ${data_father_path}")

#set(CUDA_LAUNCH_BLOCKING 1)
#set(TORCH_USE_CUDA_DSA 1)

configure_file(
        "${PROJECT_SOURCE_DIR}/include/DEFINE.h.in"
        "${PROJECT_SOURCE_DIR}/include/DEFINE.h"
)
configure_file(
        "${PROJECT_SOURCE_DIR}/index/include/Parameter.h.in"
        "${PROJECT_SOURCE_DIR}/index/include/Parameter.h"
)

#execute_process(COMMAND "conda deactivate")
#find_package(PythonInterp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED)

#link_libraries("${TORCH_LIBRARIES}" Python3::Python Python3::NumPy)
link_libraries("${TORCH_LIBRARIES}")

#set(MY_STACK_SIZE 1048576k)
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MY_STACK_SIZE}")

add_executable(main main.cpp
#        index/include/Configuration.hpp
#        index/include/Parameter.h index/include/Controller.hpp
#        index/include/experience.hpp index/include/Index.hpp
#        index/include/RL_network.hpp index/workspace/gen_exp.hpp
#        index/workspace/global_on_policy.cpp index/workspace/train.cpp
        )
# main.cpp pulls in matplotlibcpp.h (a python binding), so it needs the interpreter.
target_link_libraries(main PRIVATE Python3::Python Python3::NumPy)

add_executable(train index/workspace/train.cpp)

add_executable(train_TSMDP index/workspace/train_TSMDP.cpp)

add_executable(train_DARE index/workspace/train_DARE.cpp)

add_executable(dynamic_data benchmark/dynamic_data.cpp)

add_executable(dynamic_data2 benchmark/dynamic_data2.cpp)

# add_get_erase.cpp and Ablation.cpp both define main(); they must be separate targets.
add_executable(add_get_erase benchmark/add_get_erase.cpp)

add_executable(read_only benchmark/read_only.cpp)

add_executable(pilot_leaf benchmark/pilot_leaf.cpp)

add_executable(pilot_range benchmark/pilot_range.cpp)

# benchmark/test.cpp includes ../others/other_indexes.h. The b+tree/alex/pgm/lipp
# baselines are not part of this repository, so this target cannot be built.
# add_executable(test benchmark/test.cpp )

add_executable(opration_latency benchmark/opration_latency.cpp)

add_executable(scalability benchmark/scalability.cpp)
add_executable(Ablation benchmark/Ablation.cpp)

find_package(Threads REQUIRED)
add_executable(concurrency_test benchmark/concurrency_test.cpp)
target_link_libraries(concurrency_test PUBLIC Threads::Threads)
