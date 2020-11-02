//
// Created by Danyal Mohaddes on 2020-02-28.
//

#include <math.h>
#include <iostream>
#include <iomanip>
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

    // IO
    {
        const auto IO_ = toml::find(data, "IO");
        verbose = toml::find(IO_, "verbose").as_boolean();
        output_interval = toml::find(IO_, "output_interval").as_integer();
        output_path = toml::find(IO_, "output_path").as_string();
        output_species = toml::get<std::vector<std::string>>(toml::find(IO_, "output_species"));
    }

    // Mesh
    {
        const auto Mesh_ = toml::find(data, "Mesh");

        // Space
        N = toml::find(Mesh_, "Space", "N").as_integer();
        L = toml::find(Mesh_, "Space", "L").as_floating();
        spacing = toml::find(Mesh_, "Space", "spacing").as_string();
        if (spacing == "geometric"){
          spacing_D0 = toml::find(Mesh_, "Space", "wall_spacing").as_floating();
        }

        // Time
        time_max = toml::find(Mesh_, "Time", "time_max").as_floating();
        iteration_max = toml::find(Mesh_, "Time", "iteration_max").as_integer();
        dt = toml::find(Mesh_, "Time", "dt").as_floating();
    }

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

    // Physics
    {
        const auto Physics_ = toml::find(data, "Physics");
        m = toml::find(Physics_, "m").as_integer();
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
        av_Zl = toml::find(Numerics_, "av_Zl").as_floating();
        av_md = toml::find(Numerics_, "av_md").as_floating();
    }

    // Gas
    {
        const auto Gas_ = toml::find(data, "Gas");
        mech_file = toml::find(Gas_, "mech_file").as_string();
        phase_name = toml::find(Gas_, "phase_name").as_string();
        mech_qss = toml::find(Gas_, "qss").as_boolean();
        reacting = toml::find(Gas_, "reacting").as_boolean();
        X_ox = toml::find(Gas_, "oxidizer").as_string();
        X_f = toml::find(Gas_, "fuel").as_string();
    }

    // Spray
    {
        const auto Spray_ = toml::find(data, "Spray");
        evaporating = toml::find(Spray_,"evaporating").as_boolean();
        if (evaporating)
            X_liq = toml::find(Spray_, "species").as_string();
    }

    // BCs
    {
        const auto BCs_ = toml::find(data, "BCs");
        // Inlet
        {
            const auto Inlet_ = toml::find(BCs_, "Inlet");
            // Gas
            const auto Gas_ = toml::find(Inlet_, "Gas");
            inlet_type = toml::find(Gas_, "type").as_string();
            if (inlet_type == "mdot") {
                T_in = toml::find(Gas_, "T").as_floating();
                X_in = toml::find(Gas_, "X").as_string();
                mdot = toml::find(Gas_, "mdot").as_floating();
            } else {
                std::cerr << "Unknown Inlet BC type " << inlet_type << "not supported" << std::endl;
            }
            // Spray
            const auto Spray_ = toml::find(Inlet_, "Spray");
            Z_l_in = toml::find(Spray_, "Z_l").as_floating();
            m_d_in = toml::find(Spray_, "m_d").as_floating();
        }

        // Wall
        {
            const auto Wall_ = toml::find(BCs_, "Wall");
            // Gas
            const auto Gas_ = toml::find(Wall_, "Gas");
            wall_type = toml::find(Gas_, "type").as_string();
            if (wall_type == "adiabatic") {
                T_wall = -1.0;
            } else if (wall_type == "isothermal") {
                T_wall = toml::find(Gas_, "T").as_floating();
            } else {
                std::cerr << "Unknown Wall BC type " << wall_type << "not supported" << std::endl;
                throw(0);
            }
            // Spray
            const auto Spray_ = toml::find(Wall_, "Spray");
            filming = toml::find(Spray_, "filming").as_boolean();
        }

        // System
        p_sys = toml::find(BCs_, "System", "p").as_floating();
    }

    // ICs
    {
        const auto ICs_ = toml::find(data, "ICs");
        // Gas
        const auto Gas_ = toml::find(ICs_,"Gas");
            IC_type = toml::find(Gas_, "type").as_string();
            if (IC_type == "linear_T") {
                Tgas_0 = -1.0;
                if (wall_type == "adiabatic"){
                    std::cerr << "IC_type " << IC_type << " and wall_type " << wall_type << " not a permitted combination" << std::endl;
                }
            } else if (IC_type == "constant_T") {
                Tgas_0 = toml::find(Gas_, "T").as_floating();
            } else {
                std::cerr << "Unknown IC type " << IC_type << "not supported" << std::endl;
                throw(0);
            }
            X_0 = toml::find(Gas_, "X").as_string();
        // Spray
        const auto Spray_ = toml::find(ICs_,"Spray");
            Z_l_0 = toml::find(Spray_, "Z_l").as_floating();
            m_d_0 = toml::find(Spray_, "m_d").as_floating();
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
      if (std::strcmp(argv[i],"-Z_l_in") == 0){
        Z_l_in = atof(argv[i+1]);
        std::cout << "  Z_l_in overriden via command line to " << Z_l_in << std::endl;
      }
      if (std::strcmp(argv[i],"-m_d_in") == 0){
        m_d_in = atof(argv[i+1]);
        std::cout << "  m_d_in overriden via command line to " << m_d_in << std::endl;
      }
      if (std::strcmp(argv[i],"-T_wall") == 0){
        T_wall = atof(argv[i+1]);
        std::cout << "  T_wall overriden via command line to " << T_wall << std::endl;
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
  if (time_scheme == "CVODE") {
    // Steps from Sec. 4.4 of CVODE User Guide (V2.7.0)
    double t0 = 0.0; // initial time

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

    // Gas object
    gas = newPhase(mech_file, phase_name);

    // Kinetics
    if (mech_qss){
      gas_qss = newPhase(mech_file, "QSS");
      std::vector<ThermoPhase *> phases_{gas, gas_qss};

      kin = new GasQSSKinetics();
      importKinetics(gas_qss->xml(), phases_, kin);

    } else {
      std::vector<ThermoPhase *> phases_{gas};
      kin = newKineticsMgr(gas->xml(), phases_);
    }

    // Transport properties
    trans = newDefaultTransportMgr(gas);

    M = m + gas->nSpecies();

    if (verbose) {
        gas->setState_TPX(T_in, p_sys, X_0); // TODO this comes out as X_in ?!?
        std::cout << "  SetupGas() at T = " << gas->temperature() << " and p = " << gas->pressure()
                  << " gives viscosity = " << trans->viscosity() << " for X = " << X_0 << std::endl;
    }
}

void Solver::SetBCs() {
  // Wall
  for (int k = 0; k < M; k++){
    switch (k){
      //V
      case 0:
        wall_BC(k) = 0.0;
        break;
      // T
      case 1:
        if (wall_type == "isothermal")
          wall_BC(k) = T_wall;
        else if (wall_type == "adiabatic")
          // 1st order one-sided difference
          wall_BC(k) = phi(0,k);
        else
          throw(0);
        break;
      // TODO update Z_l and m_d BCs for filming/rebound
      // Z_l
      case 2:
        // 1st order one-sided difference in case of AV
        wall_BC(k) = phi(0,k);
        break;
      // m_d
      case 3:
        // 1st order one-sided difference in case of AV
        wall_BC(k) = phi(0,k);
        break;
      // Species
      default:
        // Species have no flux at wall for now... change when multiphase and filming
        wall_BC(k) = phi(0,k);
    }
  }

  // Inlet
  for (int k = 0; k < M; k++){
    switch (k){
      //V
      case 0:
        inlet_BC(k) = 0.0;
        break;
        // T
      case 1:
        inlet_BC(k) = T_in;
        break;
        // Z_l
      case 2:
        if (evaporating)
          inlet_BC(k) = Z_l_in;
        else
          inlet_BC(k) = 0.0;
        break;
        // m_d
      case 3:
        if (evaporating)
          inlet_BC(k) = m_d_in;
        else
          inlet_BC(k) = 1.0e-300;
        break;
        // Species
      default:
        inlet_BC(k) = Y_in(k-m);
    }
  }

  // Global strain rate
  SetState(inlet_BC);
  rho_inf = gas->density();
  double u_inf_ = mdot/rho_inf;
  a = u_inf_/L;

}

void Solver::DerivedParams() {
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
    output_header = "TITLE = \"" + input_name + "\"";
    output_header += "\nVARIABLES = \"X\", \"u\", \"ZBilger\", \"RHO\", \"V\", \"T\", \"Zl\", \"md\",";
    for (int i = 0; i < gas->nSpecies(); i++){
        output_header += " \"Y_" + gas->speciesName(i) + "\"";
        if (i != gas->nSpecies() - 1) output_header += ",";
    }
    output_header += "\nZONE I=" + std::to_string(N) + ", F=POINT";

    ign_header = "row_index,iteration,time,x,dx_avg,u,ZBilger,rho,V,T,Zl,md"; // TODO add the overridden parameters, since these are the parameter study parameters and are useful to have in the param study's (single) ignition file. Can still infer from row_index and order in which program&params were called in Python.
    for (const auto& s : output_species){
      ign_header += ",Y_" + s;
    }

    // Map of pointer to mass fractions array
    Map<const VectorXd> md_(gas->massFractions(), gas->nSpecies());

    // Initial mass fractions
    gas->setState_TPX(T_in,p_sys,X_0); // choice of temperature shouldn't make a difference for computing mass fractions here
    Y_0 = md_;

    // Inlet mass fractions
    gas->setState_TPX(T_in,p_sys,X_in);
    Y_in = md_;

    // Mixture diffusion coefficients (mass-based by default)
    mix_diff_coeffs.resize(gas->nSpecies());

    // Molar production rates
    omega_dot_mol.resize(gas->nSpecies());

    // Species molar enthalpies
    species_enthalpies_mol.resize((gas->nSpecies()));

    // Spray parameters
    // TODO change for multicomponent spray
    // TODO assuming saturated liquid for now
    if (evaporating) {
        T_l = CoolProp::PropsSI("T", "P", p_sys, "Q", 0.0, GetCoolPropString(X_liq));
        L_v = CoolProp::PropsSI("HMASS", "P", p_sys, "Q", 1.0, GetCoolPropString(X_liq)) -
              CoolProp::PropsSI("HMASS", "P", p_sys, "Q", 0.0, GetCoolPropString(X_liq));
        rho_l = CoolProp::PropsSI("DMASS", "P", p_sys, "Q", 0.0, GetCoolPropString(X_liq));
        fuel_idx = GetSpeciesIndex(X_liq);
    } else {
        T_l = L_v = rho_l = 0.0;
        fuel_idx = -1;
    }
}

std::string Solver::GetCoolPropName(const std::string cantera_name){
    std::map<std::string, std::string> cantera_to_CoolProp = {
            {"N-C12H26","NC12H26"},
            {"KERO_LUCHE","NC12H26"}
    };
    // check if it's in the map, if not throw an error. CoolProp fails silently if it doesn't recognize the name
    if (cantera_to_CoolProp.count(cantera_name)){
        return cantera_to_CoolProp[cantera_name];
    } else {
        std::cerr << "No known conversion of Cantera species " << cantera_name << " to a CoolProp species." << std::endl;
        throw(0);
    }
}

std::string Solver::GetCoolPropString(std::string cantera_string){
    std::string cool_prop_name = "";
    // Start with something like "A:0.7,B:0.3"
    std::vector<std::string> name_val_pairs;
    boost::split(name_val_pairs, cantera_string, [](char c){return c == ',';});
    // Now have ["A:0.7","B:0.3"]
    for (auto& pair : name_val_pairs){
        // Start with "A:0.7"
        std::vector<std::string> split_pair;
        boost::split(split_pair, pair, [](char c){return c == ':';});
        // Now have ["A","0.7"]
        split_pair[0] = GetCoolPropName(split_pair[0]);
        // Now have ["a","0.7"]
        pair = split_pair[0] + "[" + split_pair[1] + "]";
        // Now have "a[0.7]"
        cool_prop_name += pair + "&";
    }
    cool_prop_name = cool_prop_name.substr(0,cool_prop_name.length()-1);
    return cool_prop_name;
}

int Solver::GetSpeciesIndex(std::string cantera_string){
    // TODO assumes single component fuel!!!
    // Start with "A:0.7"
    std::vector<std::string> split_pair;
    boost::split(split_pair, cantera_string, [](char c){return c == ':';});
    // Now have ["A","0.7"]
    return gas->speciesIndex(split_pair[0]);
}

void Solver::ConstructMesh() {
    std::cout << "Solver::ConstructMesh()" << std::endl;

    // TODO make this polymorphic

    /*
     *      MESH SETUP: THERE ARE N+2 NODES, OF WHICH 2 ARE BCS
     *
     *      WALL                                                INLET
     *      |---> +x
     *
     *      | |----------|----------|----------| ... |----------| |
     *       0   dx[0]   1  dx[1]   2   dx[2]        N  dx[N]   N+1
     *
     */

    // resize vectors
    dx = VectorXd::Zero(N+1);
    nodes = VectorXd::Zero(N+2);

    if (spacing == "constant"){
      double dx_ = L/(N+1);
      dx = dx_*VectorXd::Constant(N+1,1.0);
    } else if (spacing == "geometric"){
      dx(0) = spacing_D0;
      double r_ = GetGeometricRatio<double>(N+1,L,spacing_D0);
      std::cout << "  Ratio: " << r_ << std::endl;
      for (int i=1; i<N+1; i++){
        dx(i) = pow(r_, i) * spacing_D0;
      }
      std::cout << "  Max spacing: " << dx(N)*1000.0 << "mm" << std::endl;
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

    // resize matrix
    phi = MatrixXd::Zero(N,M);

    // resize BCs
    wall_BC = RowVectorXd::Zero(M);
    inlet_BC = RowVectorXd::Zero(M);
}

void Solver::ConstructOperators() {
    std::cout << "Solver::ConstructOperators()" << std::endl;

    // TODO make this polymorphic

    // Matrices are N x N+2 such that [ddx][wall_BC , phi, inlet_BC]^T = dphi/dx, N x 1.

    // ddx
    // 1st-order 'upwinded' (but downwinded on the grid because convection is always in -ve x direction)
    // generalized to non-uniform grids

    // resize matrix
    ddx = MatrixXd::Zero(N,N+2);

    // fill matrix
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N+2; j++){
            if (i+1 == j){
                ddx(i,j)   = -1.0/dx(i+1);
                ddx(i,j+1) =  1.0/dx(i+1);
            }
        }
    }

    // d2dx2
    // 2nd-order central
    // generalized to non-uniform grids (my derivation)

    // resize matrix
    d2dx2 = MatrixXd::Zero(N,N+2);

    // fill matrix
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N+2; j++){
            if (i+1 == j){
                d2dx2(i,j-1) = (4.0 * dx(i+1))/((dx(i)*dx(i) + dx(i+1)*dx(i+1))*(dx(i)+dx(i+1)));
                d2dx2(i,j) = -4.0/(dx(i)*dx(i) + dx(i+1)*dx(i+1));
                d2dx2(i,j+1) = (4.0 * dx(i))/((dx(i)*dx(i) + dx(i+1)*dx(i+1))*(dx(i)+dx(i+1)));
            }
        }
    }

    if (verbose){
        std::cout << "ddx = \n" << ddx << std::endl;
        std::cout << "d2dx2 = \n" << d2dx2 << std::endl;
    }
}

void Solver::SetIC() {
    std::cout << "Solver::SetIC()" << std::endl;

    // TODO make this polymorphic

    // loop over all variables
    for (int k = 0; k < M; k++){
        switch (k){
            // V
            case 0:
                phi.col(k) = VectorXd::Zero(N);
                break;
            // T
            case 1:
                if (IC_type == "linear_T"){
                  double T_ = T_wall;
                  for (int i = 0; i < N; i++){
                    T_ += (T_in - T_wall)/L * dx[i];
                    phi(i,k) = T_;
                  }
                } else if (IC_type == "constant_T"){
                    phi.col(k) = Tgas_0*VectorXd::Constant(N,1.0);
                } else {
                    // should have already caught this "unknown type" error
                    throw(0);
                }
                break;
            // Z_l
            case 2:
                phi.col(k) = Z_l_0*VectorXd::Constant(N,1.0);
                break;
            // m_d
            case 3:
                // Set to very small number to ensure T and Z_l spray source terms don't become undefined
                phi.col(k) = m_d_0*VectorXd::Constant(N,1.0);
                break;
            default:
                phi.col(k) = Y_0(k-m)*VectorXd::Constant(N,1.0);
        }
    }

    if (verbose){
        std::cout << "phi_0 = \n" << phi << std::endl;
    }

    // Set BCs to match ICs
    SetBCs();
}

bool Solver::CheckIgnited() {
  bool ignited_ = false;
  if (ign_cond == "T_max") {
    ignited = (phi.col(1).maxCoeff() > T_max);
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
    std::cout << "  time = " << time << "[s]" << std::endl;
    std::cout << "  wall-time-per-iteration = " << wall_time_per_output / output_interval << " [s]" << std::endl;
    if (time_scheme == "CVODE"){
      std::cout << "  ODE steps = " << cvode_nsteps << ", RHS evals = " << cvode_nRHSevals
        << ", Jac evals = " << cvode_nJacevals << ", last dt = " << cvode_last_dt << std::endl;
    }

    // Create Phi = [wall_BC, phi, inlet_BC]^T
    MatrixXd Phi(wall_BC.rows() + phi.rows() + inlet_BC.rows(), phi.cols());
    Phi << wall_BC, phi, inlet_BC;

    // Derived quantities
    VectorXd rho_ = Getrho(Phi);
    VectorXd u_(VectorXd::Zero(N+2));
    VectorXd ZBilger_(VectorXd::Zero(N+2));
    for (int i = 0; i < N+2; i++){
      u_(i) = Getu(Phi,i);
      ZBilger_(i) = GetZBilger(Phi,i);
    }

    // Standard output
    int width_ = 14;
    std::cout << std::left << std::setw(width_) << "i" << std::setw(width_) << "x [m]" << std::setw(width_) << "u [m/s]"
      << std::setw(width_) << "rho [kg/m^3]" << std::setw(width_) << "V [1/s]"
      << std::setw(width_) << "T [K]" << std::setw(width_) << "ZBilger" << std::setw(width_) << "Z_l" << std::setw(width_)
      << "m_d" << std::setw(width_);
    for (const auto& s : output_species){
      std::cout << std::left << std::setw(width_) << "Y_" + s;
    }
    std::cout << std::endl;
    for (int i = 0; i < N+2; i++){
      std::cout << std::left << std::setw(width_) << i; // i
      std::cout << std::left << std::setw(width_) << nodes(i); // x
      std::cout << std::left << std::setw(width_) << u_(i); // u
      std::cout << std::left << std::setw(width_) << rho_(i); // rho
      std::cout << std::left << std::setw(width_) << Phi(i,0); // V
      std::cout << std::left << std::setw(width_) << std::fixed << std::setprecision(1) << Phi(i,1); // T
      std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << ZBilger_(i); // ZBilger
      std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << Phi(i,2); // Z_l
      std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << Phi(i,3); // m_d
      for (const auto& s : output_species){
        std::cout << std::left << std::setw(width_) << std::scientific << std::setprecision(2) << Phi(i,m + GetSpeciesIndex(s)); // Y
      }
      std::cout << std::endl;
    }

    // File output
    std::string output_name_;
    if (ignited && run_mode == "ignition")
      output_name_ = input_name + "_" + "ign_row_" + std::to_string(row_index) + ".dat";
    else if (!ignited && run_mode == "ignition")
      output_name_ = input_name + "_" + "no_ign_row_" + std::to_string(row_index) + ".dat";
    else
       output_name_ = input_name + "_" + std::to_string(iteration) + ".dat";
    std::ofstream output_file(output_path + output_name_);
    if (output_file.is_open()){
        std::cout << "Writing " << output_name_ << std::endl;
        output_file << output_header << std::endl;
        MatrixXd outmat_(N+2, M + 4); // X u_ ZBilger_ rho_ phi
        outmat_ << nodes, u_, ZBilger_, rho_, Phi;
        output_file << outmat_ << std::endl;
        output_file.close();
    } else {
        std::cout << "Unable to open file " << output_path + output_name_ << std::endl;
    }
}

void Solver::OutputIgnition() {
  // Console output
  std::cout << "---------------------------------------------------------" << std::endl;
  std::cout << "  Solver::OutputIgnition()" << std::endl;
  int max_row_;
  double max_T_ = phi.col(1).maxCoeff(&max_row_);
  double x_ign_ = nodes(max_row_+1);
  double dx_ign_avg_ = (dx(max_row_) + dx(max_row_+1))/2.0;
  if (ign_cond == "T_max") {
    std::cout << "  max(T) = " << max_T_ << " > T_max = " << T_max << std::endl;
    std::cout << "  Ignition at x = " << x_ign_ << " where dx_avg = " << dx_ign_avg_ << std::endl;
  }
  // Write to "ignition" file
  std::string ign_file_path_ = output_path + "ignition_data.csv";
  if (!std::ifstream(ign_file_path_)){
    std::cout << "Creating ignition_data.csv" << std::endl;
    std::ofstream ign_file_(ign_file_path_);
    ign_file_ << ign_header << std::endl;
  }

  // Create Phi = [wall_BC, phi, inlet_BC]^T
  MatrixXd Phi(wall_BC.rows() + phi.rows() + inlet_BC.rows(), phi.cols());
  Phi << wall_BC, phi, inlet_BC;

  std::ofstream ign_file_(ign_file_path_, std::ios_base::app);
  if (ign_file_.is_open()){
    std::cout << "Writing to ignition_data.csv" << std::endl;
    //row_index,iteration,time, and x,dx_avg,u,ZBilger,rho,V,T,Zl,md,Y_k @ ignition location, ignition time
    ign_file_ <<
    row_index << "," <<
    iteration << "," <<
    time << "," <<
    x_ign_ << "," <<
    dx_ign_avg_ << "," <<
    Getu(Phi,max_row_+1) << "," <<
    GetZBilger(Phi,max_row_+1) << "," <<
    Getrho(phi.row(max_row_)) << "," <<
    phi(max_row_,0) << "," <<
    phi(max_row_,1) << "," <<
    phi(max_row_,2) << "," <<
    phi(max_row_,3);
    for (const auto& s : output_species){
      ign_file_ << "," << Phi(max_row_,m + GetSpeciesIndex(s));
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
}

double Solver::Getu(const Ref<const MatrixXd>& Phi_, int i){
    if (i == 0){
        // no-slip wall
        return 0.0;
    } else {
        //TODO this can be made vastly more efficient by keeping track of the integral
        VectorXd rho_vec = Getrho(Phi_.topRows(i + 1));
        VectorXd V_vec = Phi_.col(0).head(i + 1);
        return -(2.0 / rho_vec(i)) * Quadrature(rho_vec.array() * V_vec.array(), dx.head(i));
    }
}

double Solver::GetZBilger(const Ref<const MatrixXd>& Phi_, int i){
  // This function is mostly from CharlesX, PhysicsDeriveFunctor.h, Hao Wu 2017.

  /*
   *             2(Y_C - Yo_C)/W_C + (Y_H - Yo_H)/2W_H + (Y_O - Yo_O)/W_O
   * ZBilger =  -----------------------------------------------------------
   *            2(Yf_C - Yo_C)/W_C + (Yf_H - Yo_H)/2W_H + (Yf_O - Yo_O)/W_O
   */

  size_t i_C = gas->elementIndex("C");
  size_t i_H = gas->elementIndex("H");
  size_t i_O = gas->elementIndex("O");

  double W_C = gas->atomicWeight(i_C);
  double W_H = gas->atomicWeight(i_H);
  double W_O = gas->atomicWeight(i_O);

  gas->setMoleFractionsByName(X_ox);
  double Yo_C = gas->elementalMassFraction(i_C);
  double Yo_H = gas->elementalMassFraction(i_H);
  double Yo_O = gas->elementalMassFraction(i_O);

  gas->setMoleFractionsByName(X_f);
  double Yf_C = gas->elementalMassFraction(i_C);
  double Yf_H = gas->elementalMassFraction(i_H);
  double Yf_O = gas->elementalMassFraction(i_O);

  double denom_ = 2.0 * (Yf_C - Yo_C) / W_C
                    + 0.5 * (Yf_H - Yo_H) / W_H
                    + (Yf_O - Yo_O) / W_O;

  SetState(Phi_.row(i));
  double num_   = 2.0 * (gas->elementalMassFraction(i_C) - Yo_C) / W_C
                  + 0.5 * (gas->elementalMassFraction(i_H) - Yo_H) / W_H
                  + (gas->elementalMassFraction(i_O) - Yo_O) / W_O;
  return num_/denom_;
}

double Solver::Quadrature(const Ref<const VectorXd>& f_, const Ref<const VectorXd>& dx_){
    // trapezoidal rule for non-uniform mesh
    // I = 0.5*(d_0*f_0 + d_(N-2)*f_(N-1)) + 0.5*SUM_(i=0)^(N-3) (d_(i) + d_(i+1))*f_(i+1)

    long N_ = f_.rows();
    assert(N_ >= 2);
    assert(dx_.rows() == N_ - 1);
    if (N_ == 2)
        return 0.5*(dx_(0)*f_(0) + dx_(N_-2)*f_(N_-1));
    else
        return 0.5*(dx_(0)*f_(0) + dx_(N_-2)*f_(N_-1)) + 0.5*(dx_.head(N_-2) + dx_.segment(1,N_-2)).transpose()*f_.segment(1,N_-2);
}

VectorXd Solver::Getrho(const Ref<const MatrixXd>& phi){
    VectorXd rho_vec(VectorXd::Zero(phi.rows()));
    for (int i = 0; i < phi.rows(); i++){
        SetState(phi.row(i));
        rho_vec(i) = gas->density();
    }
    return rho_vec;
}

void Solver::SetState(const Ref<const RowVectorXd>& phi){
    //TODO make sure it's safe to use the raw pointer like this
    gas->setState_TPY(phi(1),p_sys,phi.tail(gas->nSpecies()).data());
}

double Solver::Getc(int k) {
    double c;
    switch (k){
        // T
        case 1:
            c = 1.0/gas->cp_mass();
            break;
        default:
            c = 1.0;
    }
    return c;
}

double Solver::Getmu(int k) {
    double mu;
    switch (k){
        // V
        case 0:
            mu = trans->viscosity();
            //mu = 1.5e-5;
            break;
        // T
        case 1:
            mu = trans->thermalConductivity();
            //mu = 0.01;
            break;
        // Z_l
        case 2:
            mu = 0.0;
            break;
        // m_d
        case 3:
            mu = 0.0;
            break;
        // Species
        default:
            mu = mix_diff_coeffs(k-m);
    }
    return mu;
}

double Solver::Getmu_av(int k) {
  // Artificial viscosity
  double mu_av;
  switch (k){
      // Z_l
    case 2:
      mu_av = av_Zl;
      break;
      // m_d
    case 3:
      mu_av = av_md;
      break;
      // Other quantities receive no artificial viscosity
    default:
      mu_av = 0.0;
  }
  return mu_av;
}

double Solver::Getomegadot(const Ref<const RowVectorXd>& phi_, int k) {
    double omegadot_;
    switch (k){
        // V
        case 0:
            omegadot_ = rho_inf * pow(a, 2) - gas->density() * pow(phi_(0), 2);
            break;
        // T: - SUM_(i = 0)^(nSpecies) h_i^molar * omegadot_i^molar
        case 1:
            if (reacting) {
                omegadot_ = -species_enthalpies_mol.dot(omega_dot_mol);
            }
            else
                omegadot_ = 0.0;
            break;
        // Z_l
        case 2:
            omegadot_ = 0.0;
            break;
        // m_d
        case 3:
            omegadot_ = 0.0;
            break;
        // Species: omegadot_i^molar * molarmass_i
        default:
            if (reacting) {
                omegadot_ = omega_dot_mol(k - m) * gas->molecularWeight(k - m);
            }
            else {
                omegadot_ = 0.0;
            }
    }
    return omegadot_;
}

void Solver::SetDerivedVars(){
    gas->getPartialMolarEnthalpies(species_enthalpies_mol.data());
    kin->getNetProductionRates(omega_dot_mol.data());
    trans->getMixDiffCoeffs(mix_diff_coeffs.data());
}

double Solver::GetGammadot(const Ref<const RowVectorXd>& phi_, int k){
    double gammadot_;
    double rho_ = gas->density();
    double T_ = phi_(1);
    double Z_l_ = phi_(2);
    double m_d_ = phi_(3);
    switch (k){
        // V
        case 0:
            gammadot_ = 0.0;
            break;
        // T: - (rho*Z_l/m_d) * (cp * (T_l - T) - L_v)
        case 1:
            gammadot_ = - (rho_ * Z_l_ / m_d_) * (gas->cp_mass() * (T_l - T_) - L_v);
            break;
        // Z_l: + (rho*Z_l/m_d)
        case 2:
            gammadot_ = rho_ * Z_l_ / m_d_;
            break;
        // m_d: + rho
        case 3:
            gammadot_ = rho_;
            break;
        // Y_k: - (rho*Z_l/m_d) * delta_{k,f}
        default:
            if (k == m + fuel_idx)
                gammadot_ = - rho_ * Z_l_ / m_d_;
            else
                gammadot_ = 0.0;
    }
    return gammadot_;
}

double Solver::Getmdot_liq(const Ref<const RowVectorXd>& phi_){
    double mdot_;
    if (evaporating){
        double rho_ = gas->density();
        double T_ = phi_(1);
        double Z_l_ = phi_(2);
        double m_d_ = phi_(3);
        // TODO single component fuel assumed!!!
        // TODO simple Heaviside function evaporation law assumed!!!
        if (T_ > T_l){
          if (m_d_ > 0.0 && Z_l_ > 0.0){
            double A_ = 6.0/M_PI * m_d_/rho_l;
            // energy analogy mass transfer Spalding number
            double B_ = gas->cp_mass()*(T_ - T_l)/L_v;
            mdot_ = -2.0*M_PI*rho_*pow(A_,1.0/3.0)*mix_diff_coeffs(fuel_idx)*log(1.0 + B_);
          } else {
            mdot_ = 0.0;
          }
        } else {
            mdot_ = 0.0;
        }
        // The if-statement below is only useful for cases where dt is the only time step, not the outer time step
//        if (m_d_ + dt*mdot_ < 0.0 || mdot_ > 0.0){
//            //mdot_ = -m_d_/dt; // this is correct to first order, may cause undershoots and WONT work for CVODE
//            mdot_ = 0.0; // this is garbage, just to get this running
//        }

        // Guard against condensation
        if (mdot_ > 0.0){
          mdot_ = 0.0;
        }
    } else {
        mdot_ = 0.0;
    }
    return mdot_;
}

MatrixXd Solver::GetRHS(double time_, const Ref<const MatrixXd>& phi_){
  // Loop on BCs
  SetBCs();

  // Create Phi = [wall_BC, phi, inlet_BC]^T
  MatrixXd Phi(wall_BC.rows() + phi_.rows() + inlet_BC.rows(), phi_.cols());
  Phi << wall_BC, phi_, inlet_BC;

  if (verbose) {
    std::cout << "GetRHS(t = " << time_ << ", phi)" << std::endl;
    std::cout << "  Phi = " << std::endl << Phi << std::endl;
  }

  // initialize vectors and matrices
  VectorXd u(VectorXd::Zero(N));
  VectorXd rho_inv(VectorXd::Zero(N));
  MatrixXd c(MatrixXd::Zero(N,M));
  MatrixXd mu(MatrixXd::Zero(N,M));
  MatrixXd mu_av(MatrixXd::Zero(N,M));
  MatrixXd omegadot(MatrixXd::Zero(N,M));
  VectorXd mdot_liq(VectorXd::Zero(N));
  MatrixXd Gammadot(MatrixXd::Zero(N,M));

  for (int i = 0; i < N; i++){
    u(i) = Getu(Phi, i+1);
    SetState(Phi.row(i+1));
    SetDerivedVars();
    rho_inv(i) = 1.0/gas->density();
    mdot_liq(i) = Getmdot_liq(Phi.row(i+1));
    for (int k = 0; k < M; k++){
      c(i, k) = Getc(k);
      mu(i, k) = Getmu(k);
      mu_av(i, k) = Getmu_av(k);
      omegadot(i, k) = Getomegadot(Phi.row(i+1), k);
      Gammadot(i,k) = GetGammadot(Phi.row(i+1), k);
    }
  }

  // TODO make AV smarter to only activate on strong gradients

  /*
   * RHS          = conv + diff + src_gas + src_spray                              (residual definition)
   * conv         = -u*ddx*Phi                                                     (convection)
   * diff         = (diag(rho_inv)*c) .* ((mu + mu_av) .* (d2dx2 * Phi))           (diffusion, as implemented)
   * diff         = (diag(rho_inv)*c) .* (ddx * (mu .* (ddx * Phi)))               (diffusion, alternative)
   * src_gas      = (diag(rho_inv)*c) .* omegadot                                  (gas source)
   * src_spray    = (diag(rho_inv)*c) .* (diag(mdot_liq) * Gammadot)               (spray source, pure fuels only)
   */
  MatrixXd conv = -1.0*u.asDiagonal() * (ddx * Phi);
  MatrixXd diff = (rho_inv.asDiagonal() * c).array() * ((mu + mu_av).array() * (d2dx2 * Phi).array());
  MatrixXd src_gas  = (rho_inv.asDiagonal() * c).array() * omegadot.array();
  MatrixXd src_spray = (rho_inv.asDiagonal() * c).array() * (mdot_liq.asDiagonal() * Gammadot).array();

  if (verbose)
    std::cout << "RHS = " << std::endl << conv + diff + src_gas + src_spray << std::endl;

  return conv + diff + src_gas + src_spray;
}

int Solver::RunSolver() {
    std::cout << "Solver::RunSolver()" << std::endl;

    iteration = 0;
    time = 0.0;
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

            // Integrate ODE
            std::chrono::time_point<std::chrono::system_clock> tic = std::chrono::system_clock::now();
            StepIntegrator();
            std::chrono::duration<double> diff = std::chrono::system_clock::now() - tic;
            wall_time_per_output += diff.count();

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