//
// Created by Danyal Mohaddes on 2020-02-28.
//

#include <math.h>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include "Solver.h"
#include "toml.hpp"
#include "CoolProp.h"
#include "boost/algorithm/string.hpp"
#include "cvode/cvode.h"
#include "cvode/cvode_dense.h"
#include "sundials/sundials_types.h"
#include "RHSFunctor.h"
#include "Meshing.h"
#include "omp.h"
#include "Liquid/CoolPropLiquid.h"
#include "Liquid/FitLiquid.h"

#define NEAR_ONE 0.9999999999

Solver::Solver() {
    std::cout << "Solver::Solver()" << std::endl;
}

void Solver::ReadParams(int argc, char* argv[]){
    std::cout << "Solver::ReadParams()" << std::endl;

    // get input file from command line
    input_file = "";
    for (int i = 0; i < argc; i++){
        if (std::strncmp(argv[i],"-i",2) == 0){
            input_file = argv[i+1];
            std::cout << "Input file: " << input_file << std::endl;
        }
    }
    if (input_file.empty()){
        std::cerr << "No input file provided." << std::endl;
        throw(0);
    }

    // parse input file
    const toml::value data = toml::parse(input_file);

    // set this-> private members

    // Run mode
    {
      const auto RunMode_ = toml::find(data, "Run_mode");
      run_mode = toml::find(RunMode_, "mode").as_string();
      if (run_mode == "ignition"){
        ign_cond = toml::find(RunMode_,"ignition_condition").as_string();
        if (ign_cond == "T_max"){
          T_max = toml::find(RunMode_,"T_max").as_floating();
        }
      }
    }

    // IO
    {
        const auto IO_ = toml::find(data, "IO");
        verbose = toml::find(IO_, "verbose").as_boolean();
        output_interval = toml::find(IO_, "output_interval").as_integer();
        output_path = toml::find(IO_, "output_path").as_string();
        output_species = toml::get<std::vector<std::string>>(toml::find(IO_, "output_species"));
    }

    // Physics
    {
      const auto Physics_ = toml::find(data, "Physics");
      p_sys = toml::find(Physics_, "p").as_floating();

      // Gas
      {
        const auto Gas_ = toml::find(Physics_, "Gas");
        mech_file = toml::find(Gas_, "mech_file").as_string();
        phase_name = toml::find(Gas_, "phase_name").as_string();
        mech_qss = toml::find(Gas_, "qss").as_boolean();
        reacting = toml::find(Gas_, "reacting").as_boolean();
        X_ox = toml::find(Gas_, "oxidizer").as_string();
        X_f = toml::find(Gas_, "fuel").as_string();
      }

      // Spray
      {
        const auto Spray_ = toml::find(Physics_, "Spray");
        spray = toml::find(Spray_, "spray").as_boolean(); // TODO change to simply "spray" true/false. There is no use case for spray with no evap.
        if (spray) {
          X_liq = toml::find(Spray_, "species").as_string();
          liq_type = toml::find(Spray_, "properties").as_string();
          spray_gas_slip = toml::find(Spray_,"spray_gas_slip").as_boolean();
        } else {
          X_liq = "";
          liq_type = "fit";
          spray_gas_slip = false;
        }
      }

      // Solid
      {
        const auto Solid_ = toml::find(Physics_, "Solid");
        conjugate = toml::find(Solid_, "conjugate").as_boolean();
        if (conjugate){
          lam_s = toml::find(Solid_, "thermal_conductivity").as_floating();
          rho_s = toml::find(Solid_, "density").as_floating();
          c_s = toml::find(Solid_, "heat_capacity").as_floating();
        }
      }
    }

    // Mesh
    {
        const auto Mesh_ = toml::find(data, "Mesh");

        // Space
        {
          const auto Space_ = toml::find(Mesh_, "Space");
          // Fluid
          {
            const auto Fluid_ = toml::find(Space_, "Fluid");
            N = toml::find(Fluid_, "N").as_integer();
            L = toml::find(Fluid_, "L").as_floating();
            spacing = toml::find(Fluid_, "spacing").as_string();
            if (spacing == "geometric") {
              spacing_D0 = toml::find(Fluid_, "wall_spacing").as_floating();
            }
          }
          // Solid
          {
            const auto Solid_ = toml::find(Space_, "Solid");
            if (conjugate) {
              N_s = toml::find(Solid_, "N").as_integer();
              L_s = toml::find(Solid_, "L").as_floating();
              spacing_s = toml::find(Solid_, "spacing").as_string();
              if (spacing_s == "geometric") {
                spacing_D0_s = toml::find(Solid_, "wall_spacing").as_floating();
              }
            }
          }
        }

        // Time
        {
          const auto Time_ = toml::find(Mesh_, "Time");
          time_max = toml::find(Time_, "time_max").as_floating();
          iteration_max = toml::find(Time_, "iteration_max").as_integer();
          dt = toml::find(Time_, "dt").as_floating();
        }
    }

    // Numerics
    {
        const auto Numerics_ = toml::find(data, "Numerics");
        time_scheme = toml::find(Numerics_, "time_scheme").as_string();
        if (time_scheme == "CVODE"){
          cvode_abstol = toml::find(Numerics_, "cvode_abstol").as_floating();
          cvode_reltol = toml::find(Numerics_, "cvode_reltol").as_floating();
          cvode_maxsteps = toml::find(Numerics_, "cvode_maxsteps").as_integer();
        }
        n_omp_threads = toml::find(Numerics_, "openMP_threads").as_integer();
        if (spray){
          av_Zl = toml::find(Numerics_, "av_Zl").as_floating();
          av_md = toml::find(Numerics_, "av_md").as_floating();
          av_Td = toml::find(Numerics_, "av_Td").as_floating();
          // numerical experiments with dodecane sprays indicate 200 is a reasonable choice
          SF_spray = toml::find_or<double>(Numerics_, "SF_spray", 200.0);
          nonvap_frac = toml::find_or<double>(Numerics_, "nonvap_frac", 0.1);
        } else {
          av_Zl = 1.0e-5;
          av_md = 1.0e-5;
          av_Td = 1.0e-5;
          SF_spray = 0.0;
          nonvap_frac = 0.1;
        }
    }

    // BCs
    {
        const auto BCs_ = toml::find(data, "BCs");
        // Inlet
        {
            const auto Inlet_ = toml::find(BCs_, "Inlet");
            // Gas
            const auto Gas_ = toml::find(Inlet_, "Gas");
            mdot = toml::find_or<double>(Gas_, "mdot", -1.0);
            u_inf = toml::find_or<double>(Gas_, "u", -1.0);
            if (mdot > 0.0 && u_inf > 0.0) {
              std::cerr << "Inlet BC: Can only provide one of mdot or u" << std::endl;
              throw (0);
            }
            T_in = toml::find(Gas_, "T").as_floating();
            X_in = toml::find(Gas_, "X").as_string();
            // Spray
            const auto Spray_ = toml::find(Inlet_, "Spray");
            if (spray) {
              Z_l_in = toml::find(Spray_, "Z_l").as_floating();
              T_d_in = toml::find(Spray_, "T_d").as_floating();
              m_d_in = toml::find_or<double>(Spray_, "m_d", -1.0);
              D_d_in = toml::find_or<double>(Spray_, "D_d", -1.0);
              if (D_d_in > 0.0 && m_d_in > 0.0) {
                std::cerr << "Inlet BC: Can only provide one of D_d or m_d" << std::endl;
                throw (0);
              }
            } else {
              Z_l_in = 0.0;
              T_d_in = 300.0;
              m_d_in = 1.0e-300;
              D_d_in = 1.0e-300;
            }
        }

        // Wall_Interior
        {
            const auto Wall_Interior_ = toml::find(BCs_, "Wall_Interior");
            // Gas
            {
              const auto Gas_ = toml::find(Wall_Interior_, "Gas");
              if (conjugate) {
                match_T = toml::find(Gas_, "match_T").as_boolean();
              } else {
                wall_type = toml::find(Gas_, "type").as_string();
                if (wall_type == "adiabatic") {
                  T_wall = -1.0;
                } else if (wall_type == "isothermal") {
                  T_wall = toml::find(Gas_, "T").as_floating();
                } else {
                  std::cerr << "Unknown Wall BC type " << wall_type << "not supported" << std::endl;
                  throw (0);
                }
              }
            }
            // Spray
            {
              const auto Spray_ = toml::find(Wall_Interior_, "Spray");
              if (spray) {
                filming = toml::find(Spray_, "filming").as_boolean();
              } else {
                filming = false;
              }

            }
            // Solid
            {
              if (conjugate) {
                const auto Solid_ = toml::find(Wall_Interior_, "Solid");
                match_q = toml::find(Solid_, "match_q").as_boolean();
              }
            }
        }

        // Wall_Exterior
        {
          if (conjugate){
            const auto Wall_Exterior_ = toml::find(BCs_, "Wall_Exterior");
            T_s_ext = toml::find(Wall_Exterior_, "T").as_floating();
          }
        }
    }

    // ICs
    {
        const auto ICs_ = toml::find(data, "ICs");

        // Check if restart file used
        std::string tmp = toml::find_or<std::string>(ICs_,"restart","");
        if (tmp.empty()) {
          // No restart file
          IC_type = toml::find(ICs_, "type").as_string();
          if (IC_type == "linear_T") {
            Tgas_0 = -1.0;
            // Linear_T and adiabatic wall not permitted
            if (wall_type == "adiabatic") {
              std::cerr << "IC_type = " << IC_type << " and wall_type = " << wall_type << " is not a permitted combination"
                        << std::endl;
              throw(0);
            }
          } else if (IC_type == "constant_T") {
            Tgas_0 = toml::find(ICs_, "T").as_floating();
            // If constant_T, then all initial temperatures must be equal
            // For conjugate HT, that means Tgas_0 and T_s_ext
            if (conjugate && (Tgas_0 != T_s_ext)){
              std::cerr << "When IC_type = " << IC_type << " and conjugate = " << conjugate << ", Tgas_0 = " << Tgas_0
                        << " and T_s_ext = " << T_s_ext << " must be equal." << std::endl;
              throw(0);
            }
            // For isothermal wall, that means Tgas_0 and T_wall
            if (wall_type == "isothermal" && (T_wall != Tgas_0)){
              std::cerr << "When IC_type = " << IC_type << " and wall_type = " << wall_type << ", Tgas_0 = " << Tgas_0
                        << " and T_wall = " << T_wall << " must be equal." << std::endl;
              throw(0);
            }
            // No consequences for adiabatic wall.
          } else {
            std::cerr << "Unknown IC type = " << IC_type << " not supported" << std::endl;
            throw (0);
          }
          // Gas
          const auto Gas_ = toml::find(ICs_, "Gas");
          X_0 = toml::find(Gas_, "X").as_string();
          // Spray
          const auto Spray_ = toml::find(ICs_, "Spray");
          if (spray){
            Z_l_0 = toml::find(Spray_, "Z_l").as_floating();
            m_d_0 = toml::find(Spray_, "m_d").as_floating();
            T_d_0 = toml::find(Spray_, "T_d").as_floating();
          } else {
            Z_l_0 = 0.0;
            m_d_0 = 1.0e-300;
            T_d_0 = 300.0;
          }

        // With restart file
        } else {
          restart_file = tmp;
        }
    }

    // Override input file using command line
    // Consider the following variables:
    // N, L, T_in, X_in, mdot, Z_l_in, m_d_in, T_wall, p_sys, X_0
    for (int i = 0; i < argc; i++){
      if (std::strcmp(argv[i],"-row_index") == 0){
        row_index = atoi(argv[i+1]);
        std::cout << "  row_index is " << row_index << std::endl;
      }
      if (std::strcmp(argv[i],"-N") == 0){
        N = atoi(argv[i+1]);
        std::cout << "  N overriden via command line to " << N << std::endl;
      }
      if (std::strcmp(argv[i],"-L") == 0){
        L = atof(argv[i+1]);
        std::cout << "  L overriden via command line to " << L << std::endl;
      }
      if (std::strcmp(argv[i],"-T_in") == 0){
        T_in = atof(argv[i+1]);
        std::cout << "  T_in overriden via command line to " << T_in << std::endl;
      }
      if (std::strcmp(argv[i],"-X_in") == 0){
        X_in = argv[i+1];
        std::cout << "  X_in overriden via command line to " << X_in << std::endl;
      }
      if (std::strcmp(argv[i],"-mdot") == 0){
        mdot = atof(argv[i+1]);
        std::cout << "  mdot overriden via command line to " << mdot << std::endl;
      }
      if (std::strcmp(argv[i],"-u_in") == 0){
        u_inf = atof(argv[i+1]);
        std::cout << "  u_in overriden via command line to " << u_inf << std::endl;
        if (mdot > 0.0 && u_inf > 0.0) {
          std::cerr << "Overriding inlet BC: Can only provide one of mdot or u_in" << std::endl;
          throw (0);
        }
      }
      if (std::strcmp(argv[i],"-Z_l_in") == 0){
        Z_l_in = atof(argv[i+1]);
        std::cout << "  Z_l_in overriden via command line to " << Z_l_in << std::endl;
      }
      if (std::strcmp(argv[i],"-m_d_in") == 0){
        m_d_in = atof(argv[i+1]);
        std::cout << "  m_d_in overriden via command line to " << m_d_in << std::endl;
        if (D_d_in > 0.0 && m_d_in > 0.0) {
          std::cerr << "Overriding inlet BC: Can only provide one of D_d or m_d" << std::endl;
          throw (0);
        }
      }
      if (std::strcmp(argv[i],"-D_d_in") == 0){
        D_d_in = atof(argv[i+1]);
        std::cout << "  D_d_in overriden via command line to " << D_d_in << std::endl;
        if (D_d_in > 0.0 && m_d_in > 0.0) {
          std::cerr << "Overriding inlet BC: Can only provide one of D_d or m_d" << std::endl;
          throw (0);
        }
      }
      if (std::strcmp(argv[i],"-T_wall") == 0){
        T_wall = atof(argv[i+1]);
        std::cout << "  T_wall overriden via command line to " << T_wall << std::endl;
        if (conjugate) {
          std::cerr << "Cannot provide T_wall for conjugate simulation" << std::endl;
          throw(0);
        }
        if (wall_type == "adiabatic") {
          std::cerr << "Cannot provide T_wall for adiabatic wall simulation" << std::endl;
          throw(0);
        }
      }
      if (std::strcmp(argv[i],"-T_s_ext") == 0){
        T_s_ext = atof(argv[i+1]);
        std::cout << "  T_s_ext overriden via command line to " << T_s_ext << std::endl;
        if (!conjugate) {
          std::cerr << "Cannot provide T_s_ext for non-conjugate simulation" << std::endl;
          throw(0);
        }
      }
      if (std::strcmp(argv[i],"-p_sys") == 0){
        p_sys = atof(argv[i+1]);
        std::cout << "  p_sys overriden via command line to " << p_sys << std::endl;
      }
      if (std::strcmp(argv[i],"-X_0") == 0){
        X_0 = argv[i+1];
        std::cout << "  X_0 overriden via command line to " << X_0 << std::endl;
      }
    }
    if (run_mode == "ignition" && row_index < 0){
      std::cerr << "Doing an ignition parameter study, but row_index not provided. Please set with -row_index" << std::endl;
      throw(0);
    }
}

void Solver::SetupSolver() {
  std::cout << "Solver::SetupSolver()" << std::endl;
  omp_set_num_threads(n_omp_threads);
  std::cout << "  Eigen::nbThreads() = " << Eigen::nbThreads() << std::endl;
  Eigen::initParallel();
  #pragma omp parallel
  {
    std::cout << "    Thread #" << omp_get_thread_num() << " reporting" << std::endl;
  }
  if (time_scheme == "CVODE") {
    // Steps from Sec. 4.4 of CVODE User Guide (V2.7.0)
    double t0 = time; // initial time

    // 2. Set problem dimensions
    cvode_N = N * M;

    // 3. Set vector of initial values
    cvode_y = N_VNew_Serial(cvode_N);
    Eigen::Map<Eigen::MatrixXd>(NV_DATA_S(cvode_y), N, M) = phi;

    // 4. Create CVODE object
    cvode_mem = NULL;
    cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);

    // 5. Initialize CVODE solver
    CheckCVODE("CVodeInit", CVodeInit(cvode_mem, cvode_RHS, t0, cvode_y));

    // 6. Specify integration tolerances
    CheckCVODE("CVodeSStolerances" ,CVodeSStolerances(cvode_mem, cvode_reltol, cvode_abstol));

    // 7. Set optional inputs
    p_rhs_functor = new RHSFunctor(N, M, this);
    CheckCVODE("CVodeSetUserData", CVodeSetUserData(cvode_mem, p_rhs_functor));
    CheckCVODE("CVodeSetMaxStep", CVodeSetMaxStep(cvode_mem, dt));
    CheckCVODE("CVodeSetMaxNumSteps", CVodeSetMaxNumSteps(cvode_mem, cvode_maxsteps));

    // 8. Attach linear solver module
    CheckCVODE("CVDense", CVDense(cvode_mem, cvode_N));
  }
}

void Solver::CheckCVODE(std::string func_name, int flag) {
  if (flag != CV_SUCCESS){
    std::cerr << func_name << "failed!" << std::endl;
    throw(0);
  }
}

void Solver::SetupGas() {
    std::cout << "Solver::SetupGas()" << std::endl;

    for (int i=0; i<omp_get_max_threads(); i++) {
      // Gas object
      gas_vec.push_back(std::unique_ptr<ThermoPhase>(newPhase(mech_file, phase_name)));

      // Kinetics
      if (mech_qss) {
        gas_qss_vec.push_back(std::unique_ptr<ThermoPhase>(newPhase(mech_file, "QSS")));
        std::vector<ThermoPhase *> phases_{gas_vec[i].get(), gas_qss_vec[i].get()};

        kin_vec.push_back(std::unique_ptr<Kinetics>(new GasQSSKinetics()));
        importKinetics(gas_qss_vec[i]->xml(), phases_, kin_vec[i].get());

      } else {
        std::vector<ThermoPhase *> phases_{gas_vec[i].get()};
        kin_vec.push_back(std::unique_ptr<Kinetics>(newKineticsMgr(gas_vec[i]->xml(), phases_)));
      }

      // Transport properties
      trans_vec.push_back(std::unique_ptr<Transport>(newDefaultTransportMgr(gas_vec[i].get())));
    }

    int thread = omp_get_thread_num();
    ThermoPhase* gas = gas_vec[thread].get();

    n_species = gas->nSpecies();

    if (verbose) {
        gas->setState_TPX(T_in, p_sys, X_in);
        std::cout << "  SetupGas() at T = " << gas->temperature() << " and p = " << gas->pressure()
                  << " gives viscosity = " << trans_vec[thread]->viscosity() << " for X = " << X_in << std::endl;
    }
}

void::Solver::SetupLiquid(){
  if (spray) {
    std::cout << "Solver::SetupLiquid()" << std::endl;
    if (liq_type == "CoolProp") {
      liq = std::unique_ptr<Liquid>(new CoolPropLiquid(X_liq));
    } else if (liq_type == "fit") {
      liq = std::unique_ptr<Liquid>(new FitLiquid(X_liq));
    } else {
      std::cerr << "Unknown liquid type '" << liq_type << "'" << std::endl;
      throw (0);
    }
  }
}

void Solver::SetBCs() {
  // Wall
  for (int k = 0; k < M; k++){
    switch (k){
      //V
      case idx_V:
        wall_interior_BC(k) = 0.0;
        break;

      // T
      case idx_T:
        if ((wall_type == "isothermal") || conjugate){
          wall_interior_BC(k) = T_wall;
        } else if (wall_type == "adiabatic") {
          // 1st order one-sided difference
          wall_interior_BC(k) = phi(0, k);
        } else {
          throw(0);
        }
        break;

      // TODO update Z_l and m_d BCs for filming/rebound
      // Z_l
      case idx_Z_l:
        // 1st order one-sided difference in case of AV
        wall_interior_BC(k) = phi(0, k);
        break;

      // m_d
      case idx_m_d:
        // 1st order one-sided difference in case of AV
        wall_interior_BC(k) = phi(0, k);
        break;

      // T_d
      case idx_T_d:
        // 1st order one-sided difference in case of AV
        wall_interior_BC(k) = phi(0, k);
        break;

      // Species
      default:
        // TODO Species have no flux at wall for now... change when multiphase and filming
        wall_interior_BC(k) = phi(0, k);
    }
  }

  // Inlet
  for (int k = 0; k < M; k++){
    switch (k){
      //V
      case idx_V:
        inlet_BC(k) = 0.0;
        break;

      // T
      case idx_T:
        inlet_BC(k) = T_in;
        break;

      // Z_l
      case idx_Z_l:
        if (spray)
          inlet_BC(k) = Z_l_in;
        else
          inlet_BC(k) = 0.0;
        break;

      // m_d
      case idx_m_d:
        if (spray)
          inlet_BC(k) = m_d_in;
        else
          inlet_BC(k) = 1.0e-300;
        break;

      // T_d
      case idx_T_d:
        if (spray)
          inlet_BC(k) = T_d_in;
        else
          inlet_BC(k) = 300.0;
        break;

      // Species
      default:
        inlet_BC(k) = Y_in(k-m);
    }
  }

  // Global strain rate
  SetState(inlet_BC);
  int thread = omp_get_thread_num();
  rho_inf = gas_vec[thread]->density();
  double u_inf_ = mdot/rho_inf;
  a = u_inf_/L;

}

void Solver::DerivedParams() {

    std::cout << "Solver::DerivedParams()" << std::endl;
    int thread = omp_get_thread_num();

    // Input file name without path
    // Start with something like "/Users/.../inputFile.in"
    std::vector<std::string> split_string_;
    boost::split(split_string_, input_file, [](char c){return c == '/';});
    // Now have ["Users","...","inputFile.in"]
    std::vector<std::string> tmp_;
    boost::split(tmp_, split_string_[split_string_.size() - 1], [](char c){return c == '.';});
    // Now have ["inputFile","in"]
    input_name = tmp_[0];

    // Header for output files
    // Gas/Spray
    output_header = "TITLE = \"" + input_name + "\"";
    output_header += "\nVARIABLES = \"X\", \"u\", \"ZBilger\", \"RHO\", \"V\", \"T\", \"Zl\", \"md\", \"Td\",";
    for (int i = 0; i < gas_vec[0]->nSpecies(); i++){
        output_header += " \"Y_" + gas_vec[0]->speciesName(i) + "\"";
        if (i != gas_vec[0]->nSpecies() - 1) output_header += ",";
    }
    output_header += "\nZONE I=" + std::to_string(N+2) + ", F=POINT";

    // Solid
    solid_output_header = "TITLE = \"" + input_name + "\"";
    solid_output_header += "\nVARIABLES = \"X_S\", \"T_S\"";
    solid_output_header += "\nZONE I=" + std::to_string(N_s+2) + ", F=POINT";

    ign_header = "row_index,iteration,time,x,dx_avg,u,ZBilger,rho,V,T,Zl,md,Td"; // TODO add the overridden parameters, since these are the parameter study parameters and are useful to have in the param study's (single) ignition file. Can still infer from row_index and order in which program&params were called in Python.
    for (const auto& s : output_species){
      ign_header += ",Y_" + s;
    }

    // Map of pointer to mass fractions array
    Map<const VectorXd> md_(gas_vec[thread]->massFractions(), gas_vec[thread]->nSpecies());

    // Initial mass fractions, if not using restart file
    if (restart_file.empty()) {
      gas_vec[thread]->setState_TPX(T_in, p_sys,
                        X_0); // choice of temperature shouldn't make a difference for computing mass fractions here
      Y_0 = md_;
    }

    // Inlet mass fractions
    gas_vec[thread]->setState_TPX(T_in,p_sys,X_in);
    Y_in = md_;

    for (int i=0; i<omp_get_max_threads(); i++) {
      // Mixture diffusion coefficients (mass-based by default)
      mix_diff_coeffs_vec.push_back(VectorXd::Zero(n_species));

      // Molar production rates
      omega_dot_mol_vec.push_back(VectorXd::Zero(n_species));

      // Species molar enthalpies
      species_enthalpies_mol_vec.push_back(VectorXd::Zero(n_species));
    }

    // Gas parameters
    gas_vec[thread]->setState_TPX(T_in,p_sys,X_in);
    rho_inf = gas_vec[thread]->density();
    if (mdot > 0.0) u_inf = mdot/rho_inf;
    else mdot = rho_inf * u_inf;

    // Spray parameters
    // TODO change for multicomponent spray
    if (spray) {
        T_l = liq->T_sat(p_sys);
        L_v = liq->L_v(T_l);
        fuel_idx = GetSpeciesIndex(X_liq);
        double mu_ = trans_vec[thread]->viscosity(); // using inlet state from above
        if (m_d_in > 0.0) D_d_in = GetDd(m_d_in, T_d_in);
        else m_d_in = (M_PI/6.0) * liq->rho_liq(T_d_in, p_sys) * pow(D_d_in, 3.0);
        // Don't allow error due to D_min to exceed nonvap_frac of initial volume, reduce outer dt if necessary
        D_min = pow((SF_spray * dt * 18.0 * mu_ / liq->rho_liq(T_d_in, p_sys)), 0.5);
        if (pow(D_min/D_d_in,3.0) > nonvap_frac){
          std::cout << "> (D_min/D_inlet)^3 = " << pow(D_min/D_d_in,3.0) << " > nonvap_frac = " << nonvap_frac << ". Reducing dt to resolve evaporation to nonvap_frac." << std::endl;
          D_min = D_d_in * pow(nonvap_frac, 1.0/3.0);
          dt = liq->rho_liq(T_d_in, p_sys) * pow(D_min, 2.0) / (18.0 * mu_) / SF_spray;
          std::cout << ">  dt = " << dt << std::endl;
        }
        std::cout << "> D_min = " << D_min << std::endl;
        std::cout << "> (D_min/D_inlet)^3 = " << pow(D_min/D_d_in,3.0) << std::endl;

    } else {
        T_l = L_v = D_min = 0.0;
        fuel_idx = -1;
    }

    // Set physics
    if (!spray_gas_slip){
      m = 5;
    } else {
      std::cerr << "Spray-gas slip not supported." << std::endl;
      throw(0);
    }
    M = m + n_species;

    // Resizing arrays
    phi = MatrixXd::Zero(N,M);
    Phi = MatrixXd::Zero(N+2, M);
    u = VectorXd::Zero(N);
    rho_inv = VectorXd::Zero(N);
    c = MatrixXd::Zero(N,M);
    mu = MatrixXd::Zero(N,M);
    mu_av = MatrixXd::Zero(N,M);
    omegadot = MatrixXd::Zero(N,M);
    mdot_liq = VectorXd::Zero(N);
    Tdot_liq_1 = VectorXd::Zero(N);
    Tdot_liq_2 = VectorXd::Zero(N);
    Gammadot = MatrixXd::Zero(N,M);
    T_s = VectorXd::Zero(N_s);
}

int Solver::GetSpeciesIndex(std::string cantera_string){
    // TODO assumes single component fuel!!!
    // Start with "A:0.7"
    std::vector<std::string> split_pair;
    boost::split(split_pair, cantera_string, [](char c){return c == ':';});
    // Now have ["A","0.7"]
    int thread = omp_get_thread_num();
    return gas_vec[thread]->speciesIndex(split_pair[0]);
}

void Solver::ConstructMesh() {
    std::cout << "Solver::ConstructMesh()" << std::endl;

    // TODO make this polymorphic

    /*
     *      GAS/SPRAY MESH SETUP: THERE ARE N+2 NODES, OF WHICH 2 ARE BCS
     *
     *      WALL_INTERIOR                                      INLET
     *      |---> +x
     *
     *      | |----------|----------|----------| ... |----------| |
     *       0   dx[0]   1  dx[1]   2   dx[2]        N  dx[N]   N+1
     *
     *
     *      SOLID MESH SETUP: THERE ARE N_s+2 NODES, OF WHICH 2 ARE BCS
     *
     *      WALL_EXTERIOR                                 WALL_INTERIOR
     *                                                   +x  <---|
     *
     *      | |----------| ... |----------|----------|----------| |
     *     Ns+1 dxs[Ns]  Ns       dxs[2]  2  dxs[1]  1  dxs[0]   0
     *
     */

    // Gas/Spray mesh
    // resize vectors
    dx = VectorXd::Zero(N+1);
    nodes = VectorXd::Zero(N+2);

    if (spacing == "constant"){
      double dx_ = L/(N+1);
      dx = dx_*VectorXd::Constant(N+1,1.0);
    } else if (spacing == "geometric"){
      dx(0) = spacing_D0;
      auto r_ = GetGeometricRatio<double>(N+1,L,spacing_D0);
      std::cout << "  Ratio: " << r_ << std::endl;
      for (int i=1; i<N+1; i++){
        dx(i) = pow(r_, i) * spacing_D0;
      }
      std::cout << "  Max spacing (gas/spray): " << dx(N)*1000.0 << "mm" << std::endl;
    }

    // loop over node vector and fill according to spacing vector
    nodes(0) = 0.0;
    nodes(N+1) = L;
    for (int i = 1; i < N+1; i++){
        nodes(i) = nodes(i-1) + dx(i-1);
    }
    if (verbose){
        std::cout << "dx = \n" << dx << std::endl;
        std::cout << "nodes = \n" << nodes << std::endl;
    }

    // resize BCs
    wall_interior_BC = RowVectorXd::Zero(M);
    inlet_BC = RowVectorXd::Zero(M);

    // Solid mesh
    if (conjugate){
      // resize vectors
      dx_s = VectorXd::Zero(N_s+1);
      nodes_s = VectorXd::Zero(N_s+2);

      if (spacing_s == "constant"){
        double dx_ = L_s/(N_s+1);
        dx_s = dx_*VectorXd::Constant(N_s+1,1.0);
      } else if (spacing_s == "geometric"){
        dx_s(0) = spacing_D0_s;
        double r_ = GetGeometricRatio<double>(N_s+1,L_s,spacing_D0_s);
        std::cout << "  Ratio: " << r_ << std::endl;
        for (int i=1; i<N_s+1; i++){
          dx_s(i) = pow(r_, i) * spacing_D0_s;
        }
        std::cout << "  Max spacing (solid): " << dx_s(N_s)*1000.0 << "mm" << std::endl;
      }

      // loop over node vector and fill according to spacing vector
      nodes_s(0) = 0.0;
      nodes_s(N_s+1) = L_s;
      for (int i = 1; i < N_s+1; i++){
        nodes_s(i) = nodes_s(i-1) + dx_s(i-1);
      }
      if (verbose){
        std::cout << "dx_s = \n" << dx_s << std::endl;
        std::cout << "nodes_s = \n" << nodes_s << std::endl;
      }

      // resize solution vector
      T_s = VectorXd::Zero(N_s);
    }
}

void Solver::ConstructOperators() {
    std::cout << "Solver::ConstructOperators()" << std::endl;

    // TODO make this polymorphic

    // Matrices are N x N+2 such that [ddx][wall_interior_BC , phi, inlet_BC]^T = dphi/dx, N x 1.

    // ddx
    // 1st-order 'upwinded' (but downwinded on the grid because convection is always in -ve x direction)
    // generalized to non-uniform grids

    // resize matrix
    ddx.resize(N,N+2);
    ddx.reserve(2*N);

    // fill matrix
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N+2; j++){
            if (i+1 == j){
                ddx.insert(i,j)   = -1.0/dx(i+1);
                ddx.insert(i,j+1) =  1.0/dx(i+1);
            }
        }
    }

    // d2dx2
    // 2nd-order central
    // generalized to non-uniform grids (from Taylor Table)

    // resize matrix
    d2dx2.resize(N,N+2);
    d2dx2.reserve(3*N);

    // fill matrix
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N+2; j++){
            if (i+1 == j){
                double a0 = 2.0/(dx(i)*(dx(i) + dx(i+1)));
                double a1 = -2.0/(dx(i) * dx(i+1));
                double a2 = 2.0/(dx(i+1)*(dx(i) + dx(i+1)));
                d2dx2.insert(i,j-1) = a0;
                d2dx2.insert(i,j) = a1;
                d2dx2.insert(i,j+1) = a2;
            }
        }
    }

    if (verbose){
        std::cout << "ddx = \n" << MatrixXd(ddx) << std::endl;
        std::cout << "d2dx2 = \n" << MatrixXd(d2dx2) << std::endl;
    }

    if (conjugate){
      // d2dx2
      // 2nd-order central
      // generalized to non-uniform grids (my derivation)

      // resize matrix
      d2dx2_s.resize(N_s,N_s+2);
      d2dx2_s.reserve(3*N_s);

      // fill matrix
      for (int i = 0; i < N_s; i++){
        for (int j = 0; j < N_s+2; j++){
          if (i+1 == j){
            d2dx2_s.insert(i,j-1) = (4.0 * dx_s(i+1))/((dx_s(i)*dx_s(i) + dx_s(i+1)*dx_s(i+1))*(dx_s(i)+dx_s(i+1)));
            d2dx2_s.insert(i,j) = -4.0/(dx_s(i)*dx_s(i) + dx_s(i+1)*dx_s(i+1));
            d2dx2_s.insert(i,j+1) = (4.0 * dx_s(i))/((dx_s(i)*dx_s(i) + dx_s(i+1)*dx_s(i+1))*(dx_s(i)+dx_s(i+1)));
          }
        }
      }

      if (verbose){
        std::cout << "d2dx2_s = \n" << MatrixXd(d2dx2_s) << std::endl;
      }
    }
}

void Solver::SetIC() {
    std::cout << "Solver::SetIC()" << std::endl;

    if (!restart_file.empty()){
      std::cout << "> Restart fluid phase from file: " << restart_file << std::endl;
      // Set IC using restart file
      // ASSUME no change in BCs NOR variables
      // Allow a change in number of points->interpolate
      // Allow a change in outer dt-> get time from auxiliary data in file

      double T_w_gas; // gas temperature at wall, only used for ensuring solid/fluid file compatibility in conjugate mode

      {
        // Open file
        std::ifstream data_stream_(restart_file);
        if (!data_stream_.is_open()) {
          std::cerr << "Unable to open file: " << restart_file << std::endl;
          throw (0);
        }
        std::string line;
        // discard first line
        std::getline(data_stream_, line);

        // count number of variables m_ from VARIABLES = ...
        int m_;
        {
          std::getline(data_stream_, line);
          // Start with something like "A:0.7 B:0.3"
          std::vector<std::string> vars;
          boost::split(vars, line, [](char c) { return c == ' '; });
          // Now have ["A:0.7", "B:0.3"]
          m_ = vars.size() - 2; // remove "VARIABLES" and "="
          assert(m_ - 4 ==
                 M); // Derived vars X u ZBilger Rho not considered, restart file must have same number of vars as current sim
        }

        // get number of grid points n_ from ZONE I= ...; assert r > 2
        int n_;
        {
          std::getline(data_stream_, line);
          // Start with something like "ZONE I=14"
          std::vector<std::string> tmp;
          boost::split(tmp, line, [](char c) { return c == ' '; });
          // Now have ["ZONE", "I=14"]
          std::vector<std::string> tmp2;
          boost::split(tmp2, tmp[1], [](char c) { return c == '='; });
          // Now have ["I", "14"]
          n_ = std::stoi(tmp2[1]);
        }

        // create Eigen matrix old_mat of size nxm from data
        MatrixXd old_mat = MatrixXd::Zero(n_, m_);
        {
          for (int i = 0; i < n_; i++) {
            std::getline(data_stream_, line);
            // Start with something like "0.0 0.5 0.6"
            std::vector<std::string> tmp;
            boost::split(tmp, line, [](char c) { return c == ' '; }, boost::token_compress_on);
            if (tmp[0].empty()) tmp.erase(tmp.begin());
            // Now have ["0.0", "0.5", "0.6"]
            for (int j = 0; j < m_; j++) {
              old_mat(i, j) = std::stod(tmp[j]);
            }
          }
        }

        if (verbose) std::cout << "Data read from file: " << std::endl << old_mat << std::endl;

        // create position vector old_nodes from old_mat.col(0)
        VectorXd old_nodes = old_mat.col(0);

        // create old_Phi from old_mat.cols(4:end)
        MatrixXd old_Phi = old_mat.rightCols(m_ - 4);
        T_w_gas = old_Phi(0,idx_T);

        // interpolate phi from old_Phi using nodes and old_nodes
        {
          VectorXd c0 = VectorXd::Zero(N);
          VectorXi i_c0 = VectorXi::Zero(N);
          // loop on new interior nodes
          for (int i = 0; i < N; i++) {
            // find old_node to the left of current node. Always available since phi is interior, and old_Phi is full domain.
            double eps = 1.0e-10; // in case restarting from identical mesh
            int j = 0;
            while (old_nodes(j) < nodes(i + 1) + eps) {
              j++;
            }
            j -= 1;
            i_c0(i) = j;
            c0(i) = -(nodes(i + 1) - old_nodes(j)) / (old_nodes(j + 1) - old_nodes(j)) + 1.0;
          }
          VectorXd c1 = VectorXd::Ones(N) - c0;
          // Create interpolation matrix
          MatrixXd A = MatrixXd::Zero(N, n_);
          for (int i = 0; i < N; i++) {
            for (int j = 0; j < n_; j++) {
              if (j == i_c0(i)) {
                A(i, j) = c0(i);
                A(i, j + 1) = c1(i);
              }
            }
          }
          phi = A * old_Phi;
        }

        // Get time from auxiliary data in file
        {
          double time_ = -1.0;
          while (std::getline(data_stream_, line)) {
            // Start with something like "DATASETAUXDATA time = "0""
            std::vector<std::string> tmp;
            boost::split(tmp, line, [](char c) { return c == ' '; });
            // Now have ["DATASETAUXDATA", "time", "=", ""0""]
            if (tmp[1] == "time") {
              std::string tmp2 = tmp[3];
              // Now have [""0""]
              tmp2.erase(
                      remove(tmp2.begin(), tmp2.end(), '\"'),
                      tmp2.end()
              );
              time_ = std::stod(tmp2);
              break;
            }
          }
          if (time_ < 0.0)
            std::cerr << "Could not determine time from restart file: " << restart_file << std::endl;
          else {
            time = time_;
            std::cout << ">  time: " << time << " [s]" << std::endl;
          }
        }

        // Close file
        data_stream_.close();

        // Set iteration based on file name
        // Start with something like "ignition_iter_100_row_0_notign.dat"
        std::vector<std::string> tmp;
        boost::split(tmp, restart_file, [](char c) { return c == '_'; });
        // Now have ["ignition", "iter", "100", "row", "0", "notign.dat"]
        auto it = std::find(tmp.begin(), tmp.end(), "iter");
        if (it != tmp.end()) {
          std::advance(it, 1);
          iteration = std::stoi(*(it));
          std::cout << ">  iteration: " << iteration << std::endl;
        } else {
          std::cerr << "Could not determine iteration from restart file: " << restart_file << std::endl;
          throw (0);
        }
      }

      // TODO add restart capability for files with conjugate HT
      // Set solid IC using solid restart file
      if (conjugate){

        // Construct solid_restart_file name
        std::string solid_restart_file_ = restart_file;
        size_t idx = solid_restart_file_.find("iter");
        assert(idx != std::string::npos);
        solid_restart_file_.insert(idx, "solid_");
        std::cout << "> Restart solid phase from file: " << solid_restart_file_ << std::endl;

        // Open file
        std::ifstream data_stream_(solid_restart_file_);
        if (!data_stream_.is_open()) {
          std::cerr << "Unable to open file: " << solid_restart_file_ << std::endl;
          throw (0);
        }
        std::string line;
        // discard first line
        std::getline(data_stream_, line);

        // count number of variables m_ from VARIABLES = ...
        int m_;
        {
          std::getline(data_stream_, line);
          // Start with something like "A:0.7 B:0.3"
          std::vector<std::string> vars;
          boost::split(vars, line, [](char c) { return c == ' '; });
          // Now have ["A:0.7", "B:0.3"]
          m_ = vars.size() - 2; // remove "VARIABLES" and "="
          assert(m_ - 1 ==
                 1); // Derived var X not considered, solid restart file must have same number of vars as current sim, namely only T_S
        }

        // get number of grid points n_ from ZONE I= ...; assert r > 2
        int n_;
        {
          std::getline(data_stream_, line);
          // Start with something like "ZONE I=14"
          std::vector<std::string> tmp;
          boost::split(tmp, line, [](char c) { return c == ' '; });
          // Now have ["ZONE", "I=14"]
          std::vector<std::string> tmp2;
          boost::split(tmp2, tmp[1], [](char c) { return c == '='; });
          // Now have ["I", "14"]
          n_ = std::stoi(tmp2[1]);
        }

        // create Eigen matrix old_mat of size nxm from data
        MatrixXd old_mat = MatrixXd::Zero(n_, m_);
        {
          for (int i = 0; i < n_; i++) {
            std::getline(data_stream_, line);
            // Start with something like "0.0 0.5 0.6"
            std::vector<std::string> tmp;
            boost::split(tmp, line, [](char c) { return c == ' '; }, boost::token_compress_on);
            if (tmp[0].empty()) tmp.erase(tmp.begin());
            // Now have ["0.0", "0.5", "0.6"]
            for (int j = 0; j < m_; j++) {
              old_mat(i, j) = std::stod(tmp[j]);
            }
          }
        }

        if (verbose) std::cout << "Data read from file: " << std::endl << old_mat << std::endl;

        // create position vector old_nodes from old_mat.col(0)
        VectorXd old_nodes = old_mat.col(0);

        // create old_T_s from old_mat.cols(4:end)
        MatrixXd old_T_s = old_mat.rightCols(m_ - 1);

        // interpolate phi from old_T_s using nodes and old_nodes
        {
          VectorXd c0 = VectorXd::Zero(N_s);
          VectorXi i_c0 = VectorXi::Zero(N_s);
          // loop on new interior nodes
          for (int i = 0; i < N_s; i++) {
            // find old_node to the left of current node. Always available since phi is interior, and old_T_s is full domain.
            double eps = 1.0e-10; // in case restarting from identical mesh
            int j = 0;
            while (old_nodes(j) < nodes_s(i + 1) + eps) {
              j++;
            }
            j -= 1;
            i_c0(i) = j;
            c0(i) = -(nodes_s(i + 1) - old_nodes(j)) / (old_nodes(j + 1) - old_nodes(j)) + 1.0;
          }
          VectorXd c1 = VectorXd::Ones(N_s) - c0;
          // Create interpolation matrix
          MatrixXd A = MatrixXd::Zero(N_s, n_);
          for (int i = 0; i < N_s; i++) {
            for (int j = 0; j < n_; j++) {
              if (j == i_c0(i)) {
                A(i, j) = c0(i);
                A(i, j + 1) = c1(i);
              }
            }
          }
          T_s = A * old_T_s;
        }

        // Get time from auxiliary data in file
        {
          double time_ = -1.0;
          while (std::getline(data_stream_, line)) {
            // Start with something like "DATASETAUXDATA time = "0""
            std::vector<std::string> tmp;
            boost::split(tmp, line, [](char c) { return c == ' '; });
            // Now have ["DATASETAUXDATA", "time", "=", ""0""]
            if (tmp[1] == "time") {
              std::string tmp2 = tmp[3];
              // Now have [""0""]
              tmp2.erase(
                      remove(tmp2.begin(), tmp2.end(), '\"'),
                      tmp2.end()
              );
              time_ = std::stod(tmp2);
              break;
            }
          }
          if (time_ < 0.0){
            std::cerr << "Could not determine time from solid restart file: " << solid_restart_file_ << std::endl;
          } else if (time != time_){
            std::cerr << "Time from solid restart file does not agree with that of the fluid restart file." << std::endl;
            throw(0);
          }
        }

        // Close file
        data_stream_.close();

        // Set inner wall temperatures using solid restart file (only necessary for conjugate HT)
        T_wall = old_T_s(0);
        //Ensure this is the same as fluid restart file wall temperature (i.e. compatible restart files)
        assert(T_wall == T_w_gas);

      }
    } else {
      std::cout << "> Restart from initial conditions" << std::endl;

      if (conjugate){
        // estimate steady-state T_wall if conjugate HT and linear_T IC
        if (IC_type == "linear_T"){
          int thread = omp_get_thread_num();
          ThermoPhase* gas_ = gas_vec[thread].get();
          Transport* trans_ = trans_vec[thread].get();
          T_wall = GetTWall<double>(gas_,trans_,T_in,T_s_ext,p_sys,X_0,L,L_s,lam_s);
          // Steady-state T_wall = Tgas_0 = T_s_ext if conjugate HT and constant_T IC
        } else if (IC_type == "constant_T") {
          T_wall = Tgas_0;
          assert(T_wall == T_s_ext);
        }
        std::cout << "> T_wall_0 = " << std::fixed << std::setprecision(1) << T_wall << std::endl;
      }
      // in non-conjugate cases, T_wall is provided explicitly or is unnecessary (i.e. adiabatic wall)

      // Gas/spray
      // loop over all variables
      for (int k = 0; k < M; k++) {
        switch (k) {
          // V
          case idx_V:
            phi.col(k) = VectorXd::Zero(N);
            break;

            // T
          case idx_T:
            if (IC_type == "linear_T") {
              double T_ = T_wall;
              for (int i = 0; i < N; i++) {
                T_ += (T_in - T_wall) / L * dx[i];
                phi(i, k) = T_;
              }
            } else if (IC_type == "constant_T") {
              phi.col(k) = Tgas_0 * VectorXd::Constant(N, 1.0);
            }
            break;

            // Z_l
          case idx_Z_l:
            phi.col(k) = Z_l_0 * VectorXd::Constant(N, 1.0);
            break;

            // m_d
          case idx_m_d:
            // Set to very small number to ensure T and Z_l spray source terms don't become undefined
            phi.col(k) = m_d_0 * VectorXd::Constant(N, 1.0);
            break;

            // T_d
          case idx_T_d:
            phi.col(k) = T_d_0 * VectorXd::Constant(N, 1.0);
            break;

            // Y_k
          default:
            phi.col(k) = Y_0(k - m) * VectorXd::Constant(N, 1.0);
        }
      }

      // Solid
      if (conjugate){
        // always linear between T_wall and T_s_ext, constant_T is just a special case of linear_T for solid b.c. no inlet
        double T_ = T_wall;
        for (int i = 0; i < N_s; i++) {
          T_ += (T_s_ext - T_wall) / L_s * dx_s[i];
          T_s(i) = T_;
        }
      }

      // Set time and iteration to initial condition
      iteration = 0;
      time = 0.0;
    }

    if (verbose){
        std::cout << "phi_0 = \n" << phi << std::endl;
        if (conjugate){
          std::cout << "T_wall_0 = " << T_wall << std::endl;
          std::cout << "T_s_0 = \n" << T_s << std::endl;
        }
    }

    // Set BCs to match ICs
    SetBCs();

    // Set q_wall to match ICs
    SetGasQWall();
}

bool Solver::CheckIgnited() {
  bool ignited_ = false;
  if (ign_cond == "T_max") {
    ignited = (phi.col(idx_T).maxCoeff() > T_max);
  }

  if (ignited){
    std::cout << " ---------------- Ignition ---------------- " << std::endl;
  }
  return ignited;
}

bool Solver::CheckStop() {
    if ((time > time_max) || (iteration > iteration_max) || (run_mode=="ignition" && CheckIgnited()))
        return true;
    else
        return false;
}

void Solver::Output() {
    // Console output
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "  Solver::Output()" << std::endl;
    std::cout << "  iteration = " << iteration << std::endl;
    std::cout << "  time = " << std::scientific << std::setprecision(2) << time << "[s]" << std::endl;
    std::cout << "  wall-time-per-iteration = " << wall_time_per_output / output_interval << " [s]" << std::endl;
    if (time_scheme == "CVODE"){
      std::cout << "  ODE steps = " << cvode_nsteps << ", RHS evals = " << cvode_nRHSevals
        << ", Jac evals = " << cvode_nJacevals << ", last dt = " << cvode_last_dt << std::endl;
    }
    std::cout << "  q_wall = " << q_wall << " [W/m2.K]" << std::endl;
    // Create Phi = [wall_interior_BC, phi, inlet_BC]^T
    MatrixXd Phi(wall_interior_BC.rows() + phi.rows() + inlet_BC.rows(), phi.cols());
    Phi << wall_interior_BC, phi, inlet_BC;

    // Derived quantities
    VectorXd rho_ = Getrho(Phi);
    VectorXd u_(VectorXd::Zero(N+2));
    VectorXd ZBilger_(VectorXd::Zero(N+2));
    VectorXd D_d_(VectorXd::Zero(N+2));
    for (int i = 0; i < N+2; i++){
      u_(i) = Getu(Phi,i);
      ZBilger_(i) = GetZBilger(Phi,i);
      if (spray) D_d_(i) = GetDd(Phi(i, idx_m_d), Phi(i, idx_T_d));
    }

    // Console output of data
    int width_ = 14;
    std::cout << std::left << std::setw(width_) << "i" << std::setw(width_) << "x [m]" << std::setw(width_) << "u [m/s]"
      << std::setw(width_) << "rho [kg/m^3]" << std::setw(width_) << "V [1/s]"
      << std::setw(width_) << "T [K]" << std::setw(width_) << "ZBilger" << std::setw(width_);
    if (spray) {
      std::cout << "Z_l" << std::setw(width_) << "D_d [m]" << std::setw(width_) << "T_d [K]" << std::setw(width_);
    }
    for (const auto& s : output_species){
      std::cout << std::left << std::setw(width_) << "Y_" + s;
    }
    std::cout << std::endl;
    for (int i = 0; i < N+2; i++){
      std::cout << std::left << std::setw(width_) << i; // i
      std::cout << std::left << std::setw(width_) << nodes(i); // x
      std::cout << std::left << std::setw(width_) << u_(i); // u
      std::cout << std::left << std::setw(width_) << rho_(i); // rho
      std::cout << std::left << std::setw(width_) << Phi(i,idx_V); // V
      std::cout << std::left << std::setw(width_) << std::fixed << std::setprecision(1) << Phi(i,idx_T); // T
      std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << ZBilger_(i); // ZBilger
      if (spray) {
        std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2)
                  << Phi(i, idx_Z_l); // Z_l
        std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << D_d_(i); // D_d
        std::cout << std::left << std::setw(width_) << std::fixed << std::setprecision(1) << Phi(i, idx_T_d); // T_d
      }
      for (const auto& s : output_species){
        std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << Phi(i,m + GetSpeciesIndex(s)); // Y
      }
      std::cout << std::endl;
    }
    if (conjugate) {
      std::cout << "T_wall = " << std::fixed << std::setprecision(1) << T_wall << " [K]" << std::endl;
      std::cout << std::left << std::setw(width_) << "i" << std::setw(width_) << "x_s [m]"
        << std::setw(width_) << "T_s [K]"  << std::endl;
      VectorXd T_s_vec_(2 + T_s.size());
      T_s_vec_ << T_wall, T_s, T_s_ext;
      for (int i = 0; i < N_s+2; i++) {
        std::cout << std::left << std::setw(width_) << i; // i
        std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << nodes_s(i); // x_s
        std::cout << std::left << std::setw(width_) << std::fixed << std::setprecision(1) << T_s_vec_(i); // T_s
        std::cout << std::endl;
      }
    }

    // File output
    // Gas/Spray
    std::string output_name_;
    if (ignited && run_mode == "ignition")
      output_name_ = input_name + "_iter_" + std::to_string(iteration) + "_row_" + std::to_string(row_index) + "_ign.dat";
    else if (!ignited && run_mode == "ignition")
      output_name_ = input_name + "_iter_" + std::to_string(iteration) + "_row_" + std::to_string(row_index) + "_notign.dat";
    else
       output_name_ = input_name + "_iter_" + std::to_string(iteration) + ".dat";
    std::ofstream output_file(output_path + output_name_);
    if (output_file.is_open()){
        std::cout << "Writing " << output_name_ << std::endl;
        output_file << output_header << std::endl;
        MatrixXd outmat_(N+2, M + 4); // X u_ ZBilger_ rho_ Phi
        outmat_ << nodes, u_, ZBilger_, rho_, Phi;
        output_file << outmat_ << std::endl;
        output_file << "DATASETAUXDATA a = \"" << a << "\"" << std::endl;
        output_file << "DATASETAUXDATA qwall = \"" << q_wall << "\"" << std::endl;
        output_file << "DATASETAUXDATA time = \"" << time << "\"" << std::endl;
        output_file.close();
    } else {
        std::cout << "Unable to open file " << output_path + output_name_ << std::endl;
        throw(0);
    }
    // Solid
    if (conjugate) {
      std::string solid_input_name_ = input_name + "_solid";
      std::string output_name_;
      if (ignited && run_mode == "ignition")
        output_name_ = solid_input_name_ + "_iter_" + std::to_string(iteration) + "_row_" + std::to_string(row_index) + "_ign.dat";
      else if (!ignited && run_mode == "ignition")
        output_name_ = solid_input_name_ + "_iter_" + std::to_string(iteration) + "_row_" + std::to_string(row_index) + "_notign.dat";
      else
        output_name_ = solid_input_name_ + "_iter_" + std::to_string(iteration) + ".dat";
      std::ofstream output_file(output_path + output_name_);
      if (output_file.is_open()){
        std::cout << "Writing " << output_name_ << std::endl;
        output_file << solid_output_header << std::endl;
        MatrixXd outmat_(N_s+2, 2); // X T_s
        VectorXd T_s_vec_(2 + T_s.size());
        T_s_vec_ << T_wall, T_s, T_s_ext;
        outmat_ << nodes_s, T_s_vec_;
        output_file << outmat_ << std::endl;
        output_file << "DATASETAUXDATA a = \"" << a << "\"" << std::endl;
        output_file << "DATASETAUXDATA qwall = \"" << q_wall << "\"" << std::endl;
        output_file << "DATASETAUXDATA time = \"" << time << "\"" << std::endl;
        output_file.close();
      } else {
        std::cout << "Unable to open file " << output_path + output_name_ << std::endl;
        throw(0);
      }
    }
}

void Solver::OutputIgnition() {
  // Console output
  std::cout << "---------------------------------------------------------" << std::endl;
  std::cout << "  Solver::OutputIgnition()" << std::endl;
  int max_row_;
  double max_T_ = phi.col(idx_T).maxCoeff(&max_row_);
  double x_ign_ = nodes(max_row_+1);
  double dx_ign_avg_ = (dx(max_row_) + dx(max_row_+1))/2.0;
  if (ign_cond == "T_max") {
    std::cout << "  max(T) = " << std::fixed << std::setprecision(1) << max_T_ << " > T_max = " << T_max << std::endl;
    std::cout << "  Ignition at x = " << std::scientific << std::setprecision(2) << x_ign_
              << " where dx_avg = " << std::scientific << std::setprecision(2) << dx_ign_avg_ << std::endl;
  }
  // Write to "ignition" file
  std::string ign_file_path_ = output_path + "ignition_data.csv";
  if (!std::ifstream(ign_file_path_)){
    std::cout << "Creating ignition_data.csv" << std::endl;
    std::ofstream ign_file_(ign_file_path_);
    ign_file_ << ign_header << std::endl;
  }

  // Create Phi_ = [wall_interior_BC, phi, inlet_BC]^T
  MatrixXd Phi_(wall_interior_BC.rows() + phi.rows() + inlet_BC.rows(), phi.cols());
  Phi_ << wall_interior_BC, phi, inlet_BC;

  std::ofstream ign_file_(ign_file_path_, std::ios_base::app);
  if (ign_file_.is_open()){
    std::cout << "Writing to ignition_data.csv" << std::endl;
    //row_index,iteration,time, and x,dx_avg,u,ZBilger,rho,V,T,Zl,md,T_d,Y_k @ ignition location, ignition time
    ign_file_ <<
              row_index << "," <<
              iteration << "," <<
              time << "," <<
              x_ign_ << "," <<
              dx_ign_avg_ << "," <<
              Getu(Phi_, max_row_ + 1) << "," <<
              GetZBilger(Phi_, max_row_ + 1) << "," <<
              Getrho(phi.row(max_row_)) << "," <<
    phi(max_row_,idx_V) << "," <<
    phi(max_row_,idx_T) << "," <<
    phi(max_row_,idx_Z_l) << "," <<
    phi(max_row_,idx_m_d) << "," <<
    phi(max_row_,idx_T_d);
    for (const auto& s : output_species){
      ign_file_ << "," << Phi_(max_row_+1, m + GetSpeciesIndex(s));
    }
    ign_file_ << std::endl;
  }
}

void Solver::StepIntegrator() {
  // TODO make polymorphic!!

  if (time_scheme == "fwd_euler"){
    // Fwd Euler
    MatrixXd RHS = GetRHS(time, phi);
    phi = phi + dt*RHS;
  } else if (time_scheme == "TVD_RK3") {
    // TVD RK3 (Gottlieb and Shu, with time evaluations from here: http://www.cosmo-model.org/content/consortium/userSeminar/seminar2006/6_advanced_numerics_seminar/Baldauf_Runge_Kutta.pdf)
    MatrixXd RHSn_ = GetRHS(time, phi);
    MatrixXd phi1_ = phi + dt * RHSn_;
    MatrixXd RHS1_ = GetRHS(time + dt, phi1_);
    MatrixXd phi2_ = (3.0 / 4.0) * phi + (1.0 / 4.0) * phi1_ + (1.0 / 4.0) * dt * RHS1_;
    MatrixXd RHS2_ = GetRHS(time + dt / 2.0, phi2_);
    phi = (1.0 / 3.0) * phi + (2.0 / 3.0) * phi2_ + (2.0 / 3.0) * dt * RHS2_;
  } else if (time_scheme == "CVODE") {
    // CVODE BDF with numerical Jacobian
    double t_solver;
    CheckCVODE("CVode", CVode(cvode_mem, time + dt, cvode_y, &t_solver, CV_NORMAL));
    phi = Eigen::Map<Eigen::MatrixXd>(NV_DATA_S(cvode_y), N, M);
    CVodeGetNumSteps(cvode_mem, &cvode_nsteps);
    CVodeGetNumRhsEvals(cvode_mem, &cvode_nRHSevals);
    CVDlsGetNumJacEvals(cvode_mem, &cvode_nJacevals);
    CVodeGetLastStep(cvode_mem, &cvode_last_dt);
  } else {
      std::cerr << "Temporal scheme " << time_scheme << " not recognized." << std::endl;
      throw(0);
  }
  SetBCs(); // this is necessary for conjugate HT and for adiabatic walls, i.e. maintaining dynamic BCs' consistency with new solution, since SetBCs() is called by GetRHS() using partial solutions
}

void Solver::StepSolid() {
  // Using q_wall from gas

  // Step solid forward with Neumann BC on gas side, Dirichlet on outside
  // TVD RK3 (Gottlieb and Shu, with time evaluations from here: http://www.cosmo-model.org/content/consortium/userSeminar/seminar2006/6_advanced_numerics_seminar/Baldauf_Runge_Kutta.pdf)
  MatrixXd RHSn_ = GetSolidRHS(time, T_s);
  MatrixXd Ts1_ = T_s + dt * RHSn_;
  MatrixXd RHS1_ = GetSolidRHS(time + dt, Ts1_);
  MatrixXd Ts2_ = (3.0 / 4.0) * T_s + (1.0 / 4.0) * Ts1_ + (1.0 / 4.0) * dt * RHS1_;
  MatrixXd RHS2_ = GetSolidRHS(time + dt / 2.0, Ts2_);
  T_s = (1.0 / 3.0) * T_s + (2.0 / 3.0) * Ts2_ + (2.0 / 3.0) * dt * RHS2_;

  // Set T_wall from solid gas-side temperature
  T_wall = T_s(0) + q_wall * dx_s(0) / lam_s; // 1st order one-sided difference
}

void Solver::SetDerivedVars(){
    int thread = omp_get_thread_num();
    ThermoPhase* gas = gas_vec[thread].get();

    gas->getPartialMolarEnthalpies(species_enthalpies_mol_vec[thread].data());
    kin_vec[thread]->getNetProductionRates(omega_dot_mol_vec[thread].data());
    trans_vec[thread]->getMixDiffCoeffs(mix_diff_coeffs_vec[thread].data());
}

void Solver::Clipping(){
  if (spray) {
    // Enforce T_d < NEAR_ONE * T_l
    if (phi.col(idx_T_d).maxCoeff() > NEAR_ONE * T_l) {
      std::cout << "  Clipping T_d at t = " << time << "s" << std::endl;

      phi.col(idx_T_d) = phi.col(idx_T_d).cwiseMin(NEAR_ONE * T_l);

      // Re-initialize CVode, since solution has changed
      if (time_scheme == "CVODE") {
        Eigen::Map<Eigen::MatrixXd>(NV_DATA_S(cvode_y), N, M) = phi;
        CheckCVODE("CVodeReInit", CVodeReInit(cvode_mem, time, cvode_y));
      }
    }
  }
}

int Solver::RunSolver() {
    std::cout << "Solver::RunSolver()" << std::endl;

    try {
        while(!CheckStop()){

            // Outputs
            if (!(iteration % output_interval)){
                Output();
                wall_time_per_output = 0.0;
                if (verbose){
                    std::cout << "iteration = " << iteration << std::endl;
                    std::cout << "phi(t = " << time << ") = \n" << phi << std::endl;
                }
            }

            // Start timer
            std::chrono::time_point<std::chrono::system_clock> tic = std::chrono::system_clock::now();

            // Evaluate rhs for m_d, T_d
            if (spray){
              SetSprayRHS();
            }

            // Integrate gas/spray-phase system
            StepIntegrator();

            // Compute the gas-side wall heat flux
            SetGasQWall();

            // Integrate solid-phase system
            if (conjugate){
              StepSolid();
            }

            // Update timer
            std::chrono::duration<double> diff_ = std::chrono::system_clock::now() - tic;
            wall_time_per_output += diff_.count();

            // Limiters
            Clipping();

            // Update counters
            iteration++;
            time += dt;
        }

        if (ignited){
          Output();
          OutputIgnition();
        }

        return 0;
    }
    catch(...){
        std::cerr << "Error in Solver::RunSolver() at iteration = " << iteration << std::endl;
        std::cerr << "Solution upon failure: phi(t = " << time << ") = \n" << phi << std::endl;
        return 1;
    }

}

int Solver::cvode_RHS(double t, N_Vector y, N_Vector ydot, void* f_data) {
  double* ydata = NV_DATA_S(y);
  double* ydotdata = NV_DATA_S(ydot);
  auto f = (RHSFunctor*) f_data;
  return f->rhs(t, ydata, ydotdata);
}

Solver::~Solver() {
    std::cout << "Solver::~Solver()" << std::endl;
    if (time_scheme == "CVODE"){
      N_VDestroy_Serial(cvode_y);
      CVodeFree(&cvode_mem);
    }
}