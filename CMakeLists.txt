cmake_minimum_required(VERSION 3.9)
project(Hiemenz_solver)

set(CMAKE_CXX_STANDARD 11)
# If using an Intel compiler, in the following line replace "-fopenmp" with "-qopenmp"
list(APPEND CMAKE_CXX_FLAGS -fopenmp)
# For OSX, uncomment the following line:
# list(APPEND CMAKE_CXX_FLAGS "-arch x86_64 -framework Accelerate")

# Modify this path to point to the directory containing the dependencies.
# If dependencies are not all in one place, modify the "xx_ROOT" locations to reflect locations.
set(DEPS /path/to/dependencies)

set(CANTERA_ROOT ${DEPS}/cantera)
set(SUNDIALS_ROOT ${DEPS}/sundials-2.7.0/build)
set(EIGEN3_ROOT ${DEPS}/eigen/3.3.7)
set(BOOST_ROOT ${DEPS}/boost-1.55)
set(TOML_ROOT ${DEPS}/toml11)
# For OSX using Clang/LLVM compiler, uncomment the following line and specify the correct path.
# set(CLANG_LLVM_ROOT /path/to/llvm)

include_directories(${CANTERA_ROOT}/include ${SUNDIALS_ROOT}/include ${EIGEN3_ROOT}/include/eigen3 ${BOOST_ROOT}/include ${TOML_ROOT})
link_directories(${CANTERA_ROOT}/lib ${SUNDIALS_ROOT}/lib)
# For OSX, comment the line above and uncomment the line below
# link_directories(${CANTERA_ROOT}/lib ${SUNDIALS_ROOT}/lib ${CLANG_LLVM_ROOT}/lib)

add_executable(Hiemenz_solver main.cpp Solver.cpp Solver.h RHSFunctor.cpp RHSFunctor.h Meshing.h Liquid/Liquid.h Liquid/FitLiquid.cpp Liquid/FitLiquid.h SolverPhysics.cpp)
target_link_libraries(Hiemenz_solver cantera sundials_cvodes sundials_ida sundials_nvecserial)
