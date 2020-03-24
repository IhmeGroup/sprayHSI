//
// Created by Danyal Mohaddes on 2020-02-28.
//

#ifndef HIEMENZ_SOLVER_SOLVER_H
#define HIEMENZ_SOLVER_SOLVER_H

#include <string>
#include <vector>
#include "cantera/thermo.h"
#include "cantera/kinetics.h"
#include "cantera/transport.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace Cantera;

class Solver {

public:
    Solver();
    ~Solver();

    void ReadParams(int argc, char* argv[]);
    void SetupGas();
    void DerivedParams();
    void ConstructMesh();
    void ConstructOperators();
    void SetIC();
    int RunSolver();

private:
    bool CheckStop();
    void Output();
    void StepIntegrator();
    void UpdateBCs();
    void Clipping();
    MatrixXd GetRHS(double time, const Ref<const MatrixXd>& phi);
    double Getu(const Ref<const MatrixXd>& phi, int i);
    double Quadrature(const Ref<const VectorXd>& rhoV_, const Ref<const VectorXd>& dx_);
    VectorXd Getrho(const Ref<const MatrixXd>& phi);
    void SetState(const Ref<const RowVectorXd>& phi);
    double Getc(int k);
    double Getmu(int k);
    double Getomegadot(const Ref<const RowVectorXd>& phi_, int k);
    double GetGammadot(const Ref<const RowVectorXd>& phi_, int k);
    double Getmdot_liq(const Ref<const RowVectorXd>& phi_);
    std::string GetCoolPropString(std::string cantera_string);
    std::string GetCoolPropName(std::string cantera_name);
    int GetSpeciesIndex(std::string cantera_string);

    /*
     * Solution tensor \\phi, NxM
     * \\(.) is a tensor, \(.) is a vector
     * \\phi = [\V, \T, \Y1, ..., \Ym, \Zl, \md]
     */
    MatrixXd phi;

    // IO
    bool verbose;
    std::string input_file;
    int iteration;
    int output_interval; //TODO could add an output interval in time as well

    // Physics
    int M;      // number of variables per node (dimensionality)            [-]
    int m;      // number of non-species variables per node (M - nSpecies)  [-]

    // Mesh
        // Space
        VectorXd nodes;
        VectorXd dx;
        int N;      // number of finite difference nodes (degrees of freedom)   [-]
        double L;   // inlet-to-wall distance                                   [m]

        // Time
        double time;
        double dt;
        double time_max;
        double iteration_max;

    // Operators
    MatrixXd ddx;
    MatrixXd d2dx2;

    // Gas
    std::string mech_file;
    std::string mech_type;
    ThermoPhase* gas;
    Kinetics* kin;
    VectorXd omega_dot_mol;
    VectorXd species_enthalpies_mol;
    bool reacting;
    Transport* trans;
    VectorXd mix_diff_coeffs;

    // Spray
    std::string X_liq;
    double T_l;
    double L_v;
    double rho_l;
    int fuel_idx;
    bool evaporating;

    // ICs
    double Tgas_0;
    std::string X_0;
    VectorXd Y_0; // derived
    double Z_l_0;
    double m_d_0;

    // BCs
        // Inlet
        std::string inlet_type;
        double T_in;
        std::string X_in;
        double mdot;
        double Z_l_in;
        double m_d_in;
        double rho_inf; // derived
        VectorXd Y_in; // derived

        // Wall
        std::string wall_type;
        double T_wall;
        bool filming;

        // System
        double p_sys;
        double a; // derived
};


#endif //HIEMENZ_SOLVER_SOLVER_H