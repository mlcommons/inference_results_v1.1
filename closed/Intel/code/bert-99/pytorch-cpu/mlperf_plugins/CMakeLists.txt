cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mlperf_plugins)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0074 NEW)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
# PyTorch binary dependency, Must specify
# CMAKE_PREFIX_PATH=/path/to/libtorch to make it work
find_package(Torch REQUIRED)

# TODO: Force IOMP, default
if(DEFINED ENV{CONDA_PREFIX})
set(OpenMP_C_INCLUDE_DIR $ENV{CONDA_PREFIX}/include)
set(OpenMP_CXX_INCLUDE_DIR $ENV{CONDA_PREFIX}/include)
set(OpenMP_ROOT $ENV{CONDA_PREFIX})
else()
set(OpenMP_ROOT /usr/local)
endif()

find_package(OpenMP REQUIRED)

OPTION(USERCP "usercp" ON) # Enabled by default
IF(USERCP)
    ADD_DEFINITIONS(-Dusercp)
ENDIF(USERCP)

# Force options here in case cmake emit /usr/local/include
add_compile_options(-Wall -Wno-unused-function -march=native -mfma -isystem ${OpenMP_CXX_INCLUDE_DIRS})

set(DNNL_CPU_RUNTIME "OMP")
add_subdirectory(onednn)

# OneDNN dependency
# Intel OpenMP denpendency

# Bert Torch inference SUT
add_library(mlperf_plugins SHARED
  csrc/init.cpp csrc/linear.cpp csrc/cpu.cpp csrc/softmax.cpp csrc/i_softmax_tpp.cpp
  csrc/i_gelu_tpp.cpp csrc/activation.cpp csrc/i_layernorm_tpp.cpp csrc/normalization.cpp)
set_property(TARGET mlperf_plugins PROPERTY CXX_STANDARD 14)

message("Extra Torch C++ flags: ${TORCH_CXX_FLAGS}")
message("Extra OpenMP C++ flags: ${OpenMP_CXX_FLAGS}")
message("Extra OpenMP C++ includes: ${OpenMP_CXX_INCLUDE_DIRS}")
message("Extra OpenMP C++ libraries: ${OpenMP_CXX_LIBRARIES}")
message("Torch inlucde directories: ${TORCH_INCLUDE_DIRS}")
message("Torch linking libraries: ${TORCH_LIBRARIES}")

if(UNIX)
  separate_arguments(OpenMP_CXX_OPTIONS UNIX_COMMAND "${OpenMP_CXX_FLAGS}")
else()
  separate_arguments(OpenMP_CXX_OPTIONS WINDOWS_COMMAND "${OpenMP_CXX_FLAGS}")
endif()

target_compile_options(mlperf_plugins BEFORE PUBLIC ${TORCH_CXX_FLAGS} ${OpenMP_CXX_OPTIONS})
target_include_directories(mlperf_plugins SYSTEM PUBLIC include ${TORCH_INCLUDE_DIRS})
target_include_directories(mlperf_plugins PRIVATE onednn/include libxsmm/include)
target_link_directories(mlperf_plugins PRIVATE ${PROJECT_SOURCE_DIR}/libxsmm/lib)
target_link_libraries(dnnl PUBLIC ${OpenMP_CXX_LIBRARIES})
target_link_libraries(mlperf_plugins "${TORCH_LIBRARIES}" dnnl ${OpenMP_CXX_LIBRARIES})
