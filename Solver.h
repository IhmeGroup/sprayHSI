//
// Created by Danyal Mohaddes on 2020-02-28.
//

#ifndef HIEMENZ_SOLVER_SOLVER_H
#define HIEMENZ_SOLVER_SOLVER_H

#include <string>
#include <vector>
#include <chrono>
#include "nvector/nvector_serial.h"
#include "cantera/thermo.h"
#include "cantera/kinetics.h"
#include "cantera/kinetics/GasQSSKinetics.h"
#include "cantera/transport.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace Cantera;

class RHSFunctor;

class Solver {

public:
    Solver();
    ~Solver();

    void ReadParams(int argc, char* argv[]);
    void SetupSolver();
    void SetupGas();
    void SetBCs();
    void DerivedParams();
    void ConstructMesh();
    void ConstructOperators();
    void SetIC();
    int RunSolver();

  MatrixXd GetRHS(double time_, const Ref<const MatrixXd>& phi_);

private:
    bool CheckStop();
    bool CheckIgnited();
    void Output();
    void OutputIgnition();
    void StepIntegrator();
    void UpdateBCs();

    double Getu(const Ref<const MatrixXd>& Phi_, int i);
    double GetZBilger(const Ref<const MatrixXd>& Phi_, int i);
    double Quadrature(const Ref<const VectorXd>& rhoV_, const Ref<const VectorXd>& dx_);
    VectorXd Getrho(const Ref<const MatrixXd>& phi);
    void SetState(const Ref<const RowVectorXd>& phi);
    double Getc(int k);
    double Getmu(int k);
    double Getmu_av(int k);
    double Getomegadot(const Ref<const RowVectorXd>& phi_, int k);
    double GetGammadot(const Ref<const RowVectorXd>& phi_, int k);
    double Getmdot_liq(const Ref<const RowVectorXd>& phi_);
    std::string GetCoolPropString(std::string cantera_string);
    std::string GetCoolPropName(std::string cantera_name);
    int GetSpeciesIndex(std::string cantera_string);
    void AdjustRHS(Ref<MatrixXd> RHS);
    void SetDerivedVars();
    void CheckCVODE(std::string func_name, int flag);

    static int cvode_RHS(double t, N_Vector y, N_Vector ydot, void *f_data);

    /*
     * Solution tensor \\phi, NxM
     * \\(.) is a tensor, \(.) is a vector
     * \\phi = [\V, \T, \Z_l, \m_d, \Y1, ..., \YN]
     */
    MatrixXd phi;

    // Computational performance
    double wall_time_per_output = 0.0;

    // IO
    bool verbose;
    std::string input_file;
    std::string input_name;
    int iteration;
    int output_interval; //TODO could add an output interval in time as well
    std::string output_path;
    std::string output_header;
    std::string ign_header; // header for ignition file
    std::vector<std::string> output_species;
    int row_index = -1; // row index in ignition file for ignition parameter studies.

    // Physics
    int M;      // number of variables per node (dimensionality)            [-]
    int m;      // number of non-species variables per node (M - nSpecies)  [-]

    // Numerics
    std::string time_scheme;
    int n_omp_threads = 1;
    double av_Zl = 0.0;
    double av_md = 0.0;
    VectorXd u;
    VectorXd rho_inv;
    MatrixXd c;
    MatrixXd mu;
    MatrixXd mu_av;
    MatrixXd omegadot;
    VectorXd mdot_liq;
    MatrixXd Gammadot;
    MatrixXd conv;
    MatrixXd diff;
    MatrixXd src_gas;
    MatrixXd src_spray;
      // CVODE
      void* cvode_mem;
      long int cvode_N;
      long int cvode_nsteps = 0;
      long int cvode_nRHSevals = 0;
      long int cvode_nJacevals = 0;
      double cvode_last_dt = 0;
      double cvode_abstol;
      double cvode_reltol;
      long int cvode_maxsteps;
      N_Vector cvode_y;
      RHSFunctor* p_rhs_functor;

    // Mesh
        // Space
        VectorXd nodes;
        VectorXd dx;
        int N;      // number of finite difference nodes (degrees of freedom), not including BCs   [-]
        double L;   // inlet-to-wall distance                                                      [m]
        std::string spacing; // spacing scheme
        double spacing_D0; // wall spacing (first dx) for geometric spacing

        // Time
        double time;
        double dt;
        double time_max;
        double iteration_max;

    // Run mode
    std::string run_mode; // "standard" or "ignition"
    std::string ign_cond; // "T_max" for now
    double T_max;
    bool ignited = false;

    // Operators
    SparseMatrix<double> ddx;
    SparseMatrix<double> d2dx2;

    // Gas
    std::string mech_file;
    std::string phase_name;
    bool mech_qss = false;
    std::unique_ptr<ThermoPhase> gas;
    std::unique_ptr<ThermoPhase> gas_qss;
    std::unique_ptr<Kinetics> kin;
    VectorXd omega_dot_mol;
    VectorXd species_enthalpies_mol;
    bool reacting;
    std::unique_ptr<Transport> trans;
    VectorXd mix_diff_coeffs;
    std::string X_ox; // oxidizer and fuel mass fractions for ZBilger
    std::string X_f;

    // Spray
    std::string X_liq;
    double T_l;
    double L_v;
    double rho_l;
    int fuel_idx;
    bool evaporating;

    // ICs
    std::string IC_type;
    double Tgas_0;
    std::string X_0;
    VectorXd Y_0; // derived
    double Z_l_0;
    double m_d_0;
    std::string restart_file;

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
        RowVectorXd inlet_BC;

        // Wall
        std::string wall_type;
        double T_wall;
        bool filming;
        RowVectorXd wall_BC;

        // System
        double p_sys;
        double a; // derived
};


#endif //HIEMENZ_SOLVER_SOLVER_H