#include <iostream>
#include "Solver.h"
#include <chrono>

using namespace Cantera;

int main(int argc, char* argv[]) {

    // Timer
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // Init solver object
    Solver spray_solver;

    // Read params
    spray_solver.ReadParams(argc, argv);

    // Setup gas object
    spray_solver.SetupGas();

    // Setup liquid object
    spray_solver.SetupLiquid();

    // Compute derived parameters
    spray_solver.DerivedParams();

    // Construct mesh
    spray_solver.ConstructMesh();

    // Construct operators
    spray_solver.ConstructOperators();

    // Set IC
    spray_solver.SetIC();

    // Setup solver
    spray_solver.SetupSolver();

    // Run solver
    if (spray_solver.RunSolver())
        std::cerr << "spray_solver.RunSolver() failed!" << std::endl;

    // Timer
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count()*1000.0 << "ms" << std::endl;

    // Done.
    return 0;
}