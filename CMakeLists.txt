cmake_minimum_required(VERSION 3.9)
project(Hiemenz_solver)

set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_CXX_FLAGS "-arch x86_64 -framework Accelerate")

set(HOME /Users/danmohad)
set(UTILITIES_ROOT ${HOME}/Utilities)

set(CANTERA_ROOT ${UTILITIES_ROOT}/Cantera_CharlesX)
set(SUNDIALS_ROOT ${UTILITIES_ROOT}/sundials-2.7.0/build)
set(EIGEN3_ROOT /usr/local/Cellar/eigen/3.3.7)
set(BOOST_ROOT /usr/local/Cellar/boost@1.55/1.55.0_1)
set(TOML_ROOT /Users/danmohad/Utilities/toml11)

set(COOLPROP_STATIC_LIBRARY true)
add_definitions(-DNO_FMTLIB)

include_directories(${CANTERA_ROOT}/include ${SUNDIALS_ROOT}/include ${EIGEN3_ROOT}/include/eigen3 ${BOOST_ROOT}/include ${TOML_ROOT} ${UTILITIES_ROOT}/CoolProp/include)
link_directories(${CANTERA_ROOT}/lib ${SUNDIALS_ROOT}/lib ${UTILITIES_ROOT}/CoolProp/build)

add_executable(Hiemenz_solver main.cpp Solver.cpp Solver.h BoundaryConditions/BCGeneric.cpp BoundaryConditions/BCGeneric.h BoundaryConditions/Inlet.cpp BoundaryConditions/Inlet.h)
target_link_libraries(Hiemenz_solver cantera sundials_cvodes sundials_ida sundials_nvecserial CoolProp)