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
#include "Liquid/Liquid.h"

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
    void SetupLiquid();
    void SetBCs();
    void DerivedParams();
    void ConstructMesh();
    void ConstructOperators();
    void SetIC();
    int RunSolver();

    MatrixXd GetRHS(double time_, const Ref<const MatrixXd>& phi_);

private:
    VectorXd GetSolidRHS(double time_, const Ref<const VectorXd>& T_s_);
    bool CheckStop();
    bool CheckIgnited();
    void Output();
    void OutputIgnition();
    void Clipping();
    void StepIntegrator();
    void StepSolid();
    void SetSprayRHS();
    double Getu(const Ref<const MatrixXd>& Phi_, int i);
    double GetZBilger(const Ref<const MatrixXd>& Phi_, int i);
    double Quadrature(const Ref<const VectorXd>& rhoV_, const Ref<const VectorXd>& dx_);
    VectorXd Getrho(const Ref<const MatrixXd>& phi);
    void SetState(const Ref<const RowVectorXd>& phi);
    void SetGasQWall();
    double Getc(const int k);
    double Getmu(const int k);
    double Getmu_av(const int k);
    double Getomegadot(const Ref<const RowVectorXd>& phi_, const int k, const int idx);
    double GetGammadot(const Ref<const RowVectorXd>& phi_, const int k, const int idx);
    double Getmdot_liq(const Ref<const RowVectorXd>& phi_, const double mdot_liq_);
    double GetDd(const double m_d_, const double T_d_);
    double GetNu(const Ref<const RowVectorXd>& phi_);
    double GetSh(const Ref<const RowVectorXd>& phi_);
    double Getf2(const Ref<const RowVectorXd>& phi_, const double mdot_liq_);
    double GetBeta(const Ref<const RowVectorXd>& phi_, const double mdot_liq_);
    double GetHM(const Ref<const RowVectorXd>& phi_, const double mdot_liq_);
    int GetSpeciesIndex(std::string cantera_string);
    void SetDerivedVars();
    void CheckCVODE(std::string func_name, int flag);
    static int cvode_RHS(double t, N_Vector y, N_Vector ydot, void *f_data);

    enum VarIndex {idx_V, idx_T, idx_Z_l, idx_m_d, idx_T_d};

    /*
     * Solution tensor \\phi, NxM
     * \\(.) is a tensor, \(.) is a vector
     * \\phi = [\V, \T, \Z_l, \m_d, \T_d, \Y1, ..., \YN]
     */
    MatrixXd phi;

    /*
     * Solid-phase solution vector, \T_s, Nx1
     */
    VectorXd T_s;

    // Wall heat flux, positive means heat flux into wall
    double q_wall;

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
    std::string solid_output_header;
    std::string ign_header; // header for ignition file
    std::vector<std::string> output_species;
    int row_index = -1; // row index in ignition file for ignition parameter studies.

    // Physics
      // Gas and Spray
      int M;      // number of variables per node (dimensionality)            [-]
      int n_species; // number of species in mechanism [-]
      int m;      // number of non-species variables per node (M - n_species)  [-]; 4 when no spray-gas slip and spray is saturated
      bool spray_gas_slip;
      double p_sys;
      double a; // derived
      std::string liq_type; // liquid properties class to be used

      // Solid
      bool conjugate;
      double lam_s; // thermal conductivity [W/m.K]
      double rho_s; // density [kg/m3]
      double c_s; // heat capacity [J/kg.K]

    // Numerics
    std::string time_scheme;
    int n_omp_threads = 1;
    double av_Zl = 0.0;
    double av_md = 0.0;
    double av_Td = 0.0;
    double D_min;
    MatrixXd Phi;
    VectorXd u;
    VectorXd rho_inv;
    MatrixXd c;
    MatrixXd mu;
    MatrixXd mu_av;
    MatrixXd omegadot;
    VectorXd mdot_liq;
    VectorXd Tdot_liq_1;
    VectorXd Tdot_liq_2;
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

          // Gas and spray
          VectorXd nodes;
          VectorXd dx;
          int N;      // number of finite difference nodes (degrees of freedom), not including BCs   [-]
          double L;   // inlet-to-wall distance                                                      [m]
          std::string spacing; // spacing scheme
          double spacing_D0; // wall spacing (first dx) for geometric spacing

          // Solid
          VectorXd nodes_s;
          VectorXd dx_s;
          int N_s;
          double L_s;
          std::string spacing_s;
          double spacing_D0_s;

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
    SparseMatrix<double> d2dx2_s;

    // Gas
    std::string mech_file;
    std::string phase_name;
    bool mech_qss = false;
    std::vector<std::unique_ptr<ThermoPhase>> gas_vec;
    std::vector<std::unique_ptr<ThermoPhase>> gas_qss_vec;
    std::vector<std::unique_ptr<Kinetics>> kin_vec;
    std::vector<std::unique_ptr<Transport>> trans_vec;
    //std::unique_ptr<ThermoPhase> gas;
    //std::unique_ptr<ThermoPhase> gas_qss;
    //std::unique_ptr<Kinetics> kin;
    std::vector<VectorXd> omega_dot_mol_vec;
    std::vector<VectorXd> species_enthalpies_mol_vec;
    bool reacting;
    //std::unique_ptr<Transport> trans;
    std::vector<VectorXd> mix_diff_coeffs_vec;
    std::string X_ox; // oxidizer and fuel mass fractions for ZBilger
    std::string X_f;

    // Spray
    std::unique_ptr<Liquid> liq;
    std::string X_liq;
    double T_l;
    double L_v;
    int fuel_idx;
    bool spray;
    const double A_ref = 1.0/3.0; // 1/3 rule

    // ICs
    std::string IC_type;
    double Tgas_0;
    std::string X_0;
    VectorXd Y_0; // derived
    double Z_l_0;
    double m_d_0;
    double T_d_0;
    std::string restart_file;

    // BCs
        // Inlet
        std::string inlet_type;
        double T_in;
        std::string X_in;
        double mdot;
        double Z_l_in;
        double m_d_in;
        double D_d_in;
        double T_d_in;
        double rho_inf; // derived
        VectorXd Y_in; // derived
        RowVectorXd inlet_BC;

        // Wall_Interior
          // Gas
          bool match_T;
          std::string wall_type;
          RowVectorXd wall_interior_BC;
          // Spray
          bool filming;
          // Solid
          bool match_q;
          double T_wall; // this is equivalent to "T_s_int"


        // Wall_Exterior
        double T_s_ext;
};


#endif //HIEMENZ_SOLVER_SOLVER_H