//
// Created by Danyal Mohaddes on 2020-02-28.
//

#include <iostream>
#include <iomanip>
#include "Solver.h"



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

    //TODO read from an input file

    // parse input file

    // set this-> private members
    verbose = false;

    N = 21;
    L = 0.1;
    time_max = 100.0;
    iteration_max = 10000;
    output_interval = 1000;
    dt = 0.01;

    M = 2;

    mech_file = "air.xml";
    mech_type = "";

    Tgas_0 = 300.0;
    T_in = 300.0;
    T_wall = 800.0;
    p_sys = 101325.0;

    mdot = 0.01;

}

void Solver::SetupGas() {
    std::cout << "Solver::SetupGas()" << std::endl;

    //TODO add trans and kin objects for transport and kinetics
    gas = newPhase(mech_file,mech_type);
    std::vector<ThermoPhase*> phases_ {gas};
    kin = newKineticsMgr(gas->xml(),phases_);
    trans = newDefaultTransportMgr(gas);

    if (verbose)
        gas->setState_TPX(300.0, 101325.0, "O2:1.0,N2:3.76");
        std::cout << "  SetupGas() at T = " << gas->temperature() << " and p = " << gas->pressure() << " gives viscosity = " << trans->viscosity() << std::endl;

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

    // basic IC; should get from input file
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
            // TODO this is a good place for an assert and error out if the case has not been implemented
            default:
                phi.col(k) = VectorXd::Zero(N);
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
                phi(0,k) = T_wall;
                break;

            default:
                phi(0,k) = 0.0;
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

            default:
                phi(N-1,k) = 0.0;
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
    //TODO change this function when species equation is added in
    gas->setState_TP(phi(1),p_sys);
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
    //TODO change this function when species equation is added in
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
        default:
            mu = 1.0;
    }
    return mu;
}

double Solver::Getomegadot(const Ref<const RowVectorXd>& phi, int k) {
    //TODO change this function when species equation is added in
    double omegadot;
    switch (k){
        // V
        case 0:
            omegadot = rho_inf*pow(a,2) - gas->density()*pow(phi(0),2);
            break;
        // T
        //TODO add temperature source term when species equation is added in
        case 1:
            omegadot = 0.0;
            break;
        default:
            omegadot = 0.0;
    }
    return omegadot;
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
     * src  = diag(rho_inv) * omegadot
     */
    MatrixXd conv = -1.0*u.asDiagonal() * (ddx * phi);
    MatrixXd diff = (rho_inv.asDiagonal() * c).array() * (mu.array() * (d2dx2 * phi).array());
    MatrixXd src  = rho_inv.asDiagonal() * omegadot;

    return conv + diff + src;
}

int Solver::RunSolver() {
    std::cout << "Solver::RunSolver()" << std::endl;

    iteration = 0;
    try {
        while(!CheckStop()){

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

            // Outputs
            if (!(iteration % output_interval)){
                Output();
            }

            if (verbose){
                std::cout << "iteration = " << iteration << std::endl;
                std::cout << "phi(t = " << time << ") = \n" << phi << std::endl;
            }
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