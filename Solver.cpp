//
// Created by Danyal Mohaddes on 2020-02-28.
//

#include <iostream>
#include <iomanip>
#include "Solver.h"
#include "toml.hpp"

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
    verbose = toml::find(data,"IO","verbose").as_boolean();
    output_interval = toml::find(data,"IO","output_interval").as_integer();

    // Mesh
    const auto mesh_ = toml::find(data,"Mesh");

        // Space
        N = toml::find(mesh_,"Space","N").as_integer();
        L = toml::find(mesh_,"Space","L").as_floating();

        // Time
        time_max = toml::find(mesh_,"Time","time_max").as_floating();
        iteration_max = toml::find(mesh_,"Time","iteration_max").as_integer();
        dt = toml::find(mesh_,"Time","dt").as_floating();

    // Physics
    m = toml::find(data,"Physics","m").as_integer();
    reacting = toml::find(data,"Physics","reacting").as_boolean();

    // Gas
    mech_file = toml::find(data,"Gas","mech_file").as_string();
    mech_type = toml::find(data,"Gas","mech_type").as_string();

    // BCs
    const auto BCs_ = toml::find(data,"BCs");

        // Inlet
        std::string inlet_type = toml::find(BCs_,"Inlet","type").as_string();
        if (inlet_type == "mdot") {
            T_in = toml::find(BCs_, "Inlet", "T").as_floating();
            X_in = toml::find(BCs_, "Inlet", "X").as_string();
            mdot = toml::find(BCs_, "Inlet", "mdot").as_floating();
        } else {
            std::cerr << "Unknown Inlet BC type " << inlet_type << "not supported" << std::endl;
        }

        // Wall
        wall_type = toml::find(BCs_,"Wall","type").as_string();
        if (wall_type == "adiabatic"){
            T_wall = -1.0;
        } else if (wall_type == "isothermal"){
            T_wall = toml::find(BCs_,"Wall","T").as_floating();
        } else {
            std::cerr << "Unknown Wall BC type " << wall_type << "not supported" << std::endl;
        }

        // System
        p_sys = toml::find(BCs_,"System","p").as_floating();

    // ICs
    Tgas_0 = toml::find(data,"ICs","T").as_floating();
    X_0 = toml::find(data,"ICs","X").as_string();
}

void Solver::SetupGas() {
    std::cout << "Solver::SetupGas()" << std::endl;

    gas = newPhase(mech_file,mech_type);
    std::vector<ThermoPhase*> phases_ {gas};
    kin = newKineticsMgr(gas->xml(),phases_);
    trans = newDefaultTransportMgr(gas);

    M = m + gas->nSpecies();

    if (verbose) {
        gas->setState_TPX(Tgas_0, p_sys, X_0);
        std::cout << "  SetupGas() at T = " << gas->temperature() << " and p = " << gas->pressure()
                  << " gives viscosity = " << trans->viscosity() << " for X = " << X_0 << std::endl;
    }
}

void Solver::DerivedParams() {
    // Map of pointer to mass fractions array
    Map<const VectorXd> md_(gas->massFractions(), gas->nSpecies());

    // Initial mass fractions
    gas->setState_TPX(Tgas_0,p_sys,X_0);
    Y_0 = md_;

    // Inlet mass fractions
    gas->setState_TPX(T_in,p_sys,X_in);
    Y_in = md_;

    // Mixture diffusion coefficients (mass-based by default)
    mix_diff_coeffs.resize(gas->nSpecies());
    trans->getMixDiffCoeffs(mix_diff_coeffs.data());

    // Molar production rates
    omega_dot_mol.resize(gas->nSpecies());
    kin->getNetProductionRates(omega_dot_mol.data());

    // Species molar enthalpies
    species_enthalpies_mol.resize((gas->nSpecies()));
    gas->getPartialMolarEnthalpies(species_enthalpies_mol.data());
}

void Solver::ConstructMesh() {
    std::cout << "Solver::ConstructMesh()" << std::endl;

    // TODO make this polymorphic

    /*
     *      MESH SETUP
     *
     *      WALL                                                INLET
     *      |---> +x
     *
     *      | |----------|----------|----------| ... |----------| |
     *       0   dx[0]   1  dx[1]   2   dx[2]       N-2  dx[N-2]  N-1
     *
     */

    // resize vectors
    dx = VectorXd::Zero(N-1);
    nodes = VectorXd::Zero(N);

    // constant spacing for now, but this could be specified from input (e.g. log spacing)
    double dx_ = L/(N-1);
    dx = dx_*VectorXd::Constant(N,1.0);

    // loop over node vector and fill according to spacing vector
    nodes(0) = 0.0;
    nodes(N-1) = L;
    for (int i = 1; i < N-1; i++){
        nodes(i) = nodes(i-1) + dx(i-1);
    }
    if (verbose){
        std::cout << "dx = \n" << dx << std::endl;
        std::cout << "nodes = \n" << nodes << std::endl;
    }
}

void Solver::ConstructOperators() {
    std::cout << "Solver::ConstructOperators()" << std::endl;

    // TODO make this polymorphic

    // ddx
    // 1st-order 'upwinded' (but downwinded on the grid because convection is always in -ve x direction)

    // resize matrix
    ddx = MatrixXd::Zero(N,N);

    // fill matrix
    for (int i = 1; i < N-1; i++){
        for (int j = 0; j < N; j++){
            if (i == j){
                ddx(i,j)   = -1.0/dx[i];
                ddx(i,j+1) =  1.0/dx[i];
            }
        }
    }

    // d2dx2
    // 2nd-order central
    //TODO this is only 2nd-order for uniform grids! must include extra terms for non-uniform or else 0th order!!

    // resize matrix
    d2dx2 = MatrixXd::Zero(N,N);

    // fill matrix
    for (int i = 1; i < N-1; i++){
        for (int j = 0; j < N; j++){
            if (i == j){
                double dx2_ = (pow(dx(i-1),2) + pow(dx(i),2))/2.0;
                d2dx2(i,j-1) =  1.0/dx2_;
                d2dx2(i,j)   = -2.0/dx2_;
                d2dx2(i,j+1) =  1.0/dx2_;
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

    // resize matrix
    phi = MatrixXd::Zero(N,M);

    // loop over all variables
    for (int k = 0; k < M; k++){
        switch (k){
            // V
            case 0:
                phi.col(k) = VectorXd::Zero(N);
                break;
            // T
            case 1:
                phi.col(k) = Tgas_0*VectorXd::Constant(N,1.0);
                break;
            // TODO add Z_l, m_d as cases here
            default:
                phi.col(k) = Y_0(k-m)*VectorXd::Constant(N,1.0);
        }
    }

    if (verbose){
        std::cout << "phi_0 = \n" << phi << std::endl;
    }
}

bool Solver::CheckStop() {
    if ((time > time_max) || (iteration > iteration_max))
        return true;
    else
        return false;
}

void Solver::Output() {
    // TODO add output to file
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "  Solver::Output()" << std::endl;
    std::cout << "  iteration = " << iteration << std::endl;
    std::cout << "  time = " << time << "[s]" << std::endl;

    int width_ = 10;
    std::cout << std::left << std::setw(width_) << "i" << std::setw(width_) << "x [m]" << std::setw(width_) << "V [1/s]" << std::setw(width_) << "T [K]" <<'\n';
    for (int i = 0; i < N; i++){
        std::cout << std::left << std::setw(width_) << i << std::setw(width_) << nodes(i) << std::setw(width_) << phi(i,0) << std::setw(width_) << phi(i,1) << std::endl;
    }
}

void Solver::UpdateBCs() {
    // TODO make this polymorphic!!
    // Isothermal wall and mdot inlet for now

    // Wall
    for (int k = 0; k < M; k++){
        switch (k){
            //V
            case 0:
                phi(0,k) = 0.0;
                break;
            // T
            case 1:
                if (wall_type == "isothermal")
                    phi(0,k) = T_wall;
                else if (wall_type == "adiabatic")
                    // 1st order one-sided difference
                    phi(0,k) = phi(1,k);
                else
                    std::cerr << "unknown wall type and this should be polymorphic anyway" << std::endl;
                break;
            // TODO update this for Z_l and m_d equations
            // Species
            default:
                // Species have no flux at wall for now... change when multiphase and filming
                phi(0,k) = phi(1,k);
        }
    }

    // Inlet
    for (int k = 0; k < M; k++){
        switch (k){
            //V
            case 0:
                phi(N-1,k) = 0.0;
                break;
            // T
            case 1:
                phi(N-1,k) = T_in;
                break;
            // TODO update this for Z_l and m_d equations
            // Species
            default:
                phi(N-1,k) = Y_in(k-m);
        }
    }

    // Global strain rate
    SetState(phi.row(N-1));
    rho_inf = gas->density();
    double u_inf_ = mdot/rho_inf;
    a = u_inf_/L;
}

void Solver::StepIntegrator() {
    // TODO make polymorphic!!
    // Fwd Euler
    MatrixXd RHS = GetRHS(time,phi);
    phi = phi + dt*RHS;
}

double Solver::Getu(const Ref<const MatrixXd>& phi, int i){
    //TODO this can be made vastly more efficient by keeping track of the integral
    VectorXd rho_vec = Getrho(phi.topRows(i+1));
    VectorXd V_vec   = phi.col(0).head(i+1);
    return -(2.0/rho_vec(i)) * Quadrature(rho_vec.array() * V_vec.array(),dx.head(i));
}

double Solver::Quadrature(const Ref<const VectorXd>& f_, const Ref<const VectorXd>& dx_){
    // trapezoidal rule for non-uniform mesh
    // I = 0.5*(d_0*f_0 + d_(N-2)*f_(N-1)) + 0.5*SUM_(i=0)^(N-3) (d_(i) + d_(i+1))*f_(i+1)

    long N_ = f_.rows();
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
        // TODO make sure to set mu = 0 for Z_l and m_d
        // Species
        default:
            mu = mix_diff_coeffs(k-m);
    }
    return mu;
}

double Solver::Getomegadot(const Ref<const RowVectorXd>& phi, int k) {
    double omegadot_;
    switch (k){
        // V
        case 0:
            omegadot_ = rho_inf * pow(a, 2) - gas->density() * pow(phi(0), 2);
            break;
        // T: - SUM_(i = 0)^(nSpecies) h_i^molar * omegadot_i^molar
        case 1:
            if (reacting)
                omegadot_ = - species_enthalpies_mol.dot(omega_dot_mol);
            else
                omegadot_ = 0.0;
            break;
        // TODO add Z_l, m_d sources here
        // Species: omegadot_i^molar * molarmass_i
        default:
            if (reacting)
                omegadot_ = omega_dot_mol(k-m)*gas->molecularWeight(k-m);
            else
                omegadot_ = 0.0;
    }
    return omegadot_;
}

MatrixXd Solver::GetRHS(double time, const Ref<const MatrixXd>& phi){

    // initialize vectors and matrices
    VectorXd u(VectorXd::Zero(N));
    VectorXd rho_inv(VectorXd::Zero(N));
    MatrixXd c(MatrixXd::Zero(N,M));
    MatrixXd mu(MatrixXd::Zero(N,M));
    MatrixXd omegadot(MatrixXd::Zero(N,M));

    for (int i = 1; i < N-1; i++){
        u(i) = Getu(phi, i);
        SetState(phi.row(i));
        rho_inv(i) = 1.0/gas->density();
        for (int k = 0; k < M; k++){
            c(i, k) = Getc(k);
            mu(i, k) = Getmu(k);
            omegadot(i, k) = Getomegadot(phi.row(i),k);
        }
    }

    /*
     * RHS = conv + diff + src
     * conv = -u*ddx*phi
     * diff = (diag(rho_inv)*c) .* (ddx * (mu .* (ddx * phi))) (alternative)
     * diff = (diag(rho_inv)*c) .* (mu .* (d2dx2 * phi))
     * src  = (diag(rho_inv)*c) .* omegadot
     */
    MatrixXd conv = -1.0*u.asDiagonal() * (ddx * phi);
    MatrixXd diff = (rho_inv.asDiagonal() * c).array() * (mu.array() * (d2dx2 * phi).array());
    MatrixXd src  = (rho_inv.asDiagonal() * c).array() * omegadot.array();

    return conv + diff + src;
}

int Solver::RunSolver() {
    std::cout << "Solver::RunSolver()" << std::endl;

    iteration = 0;
    try {
        while(!CheckStop()){

            // Outputs
            if (!(iteration % output_interval)){
                Output();
            }

            if (verbose){
                std::cout << "iteration = " << iteration << std::endl;
                std::cout << "phi(t = " << time << ") = \n" << phi << std::endl;
            }

            // Loop on BCs
            //inlet_bc.Update();
            //wall_bc.Update();
            UpdateBCs(); // non-polymorphic version for initial testing

            // Integrate ODE
            //integrator.Step();
            StepIntegrator();// non-polymorphic version for initial testing

            // Update counters
            iteration++;
            time += dt;


        }

        return 0;
    }
    catch(...){
        std::cerr << "Error in Solver::RunSolver() at iteration = " << iteration << std::endl;
        std::cerr << "Solution upon failure: phi(t = " << time << ") = \n" << phi << std::endl;
        return 1;
    }

}

Solver::~Solver() {
    std::cout << "Solver::~Solver()" << std::endl;
}