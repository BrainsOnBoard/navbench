cmake_minimum_required(VERSION 3.1)

# Clones the bob_robotics git submodule and includes the main CMake file. For
# more details on the BoB build system, see: https://github.com/BrainsOnBoard/bob_robotics/tree/cmake/cmake
include(get_submodule.cmake)
BoB_project(SOURCES benchmark_infomax.cc benchmark_pm.cc
                    benchmark_perf_vs_num_snaps.cc
            BOB_MODULES common navigation)
