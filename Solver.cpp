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
    }

    // Mesh
    {
        const auto Mesh_ = toml::find(data, "Mesh");

        // Space
        N = toml::find(Mesh_, "Space", "N").as_integer();
        L = toml::find(Mesh_, "Space", "L").as_floating();

        // Time
        time_max = toml::find(Mesh_, "Time", "time_max").as_floating();
        iteration_max = toml::find(Mesh_, "Time", "iteration_max").as_integer();
        dt = toml::find(Mesh_, "Time", "dt").as_floating();
    }

    // Physics
    {
        const auto Physics_ = toml::find(data, "Physics");
        m = toml::find(Physics_, "m").as_integer();
    }

    // Gas
    {
        const auto Gas_ = toml::find(data, "Gas");
        mech_file = toml::find(Gas_, "mech_file").as_string();
        mech_type = toml::find(Gas_, "mech_type").as_string();
        reacting = toml::find(Gas_, "reacting").as_boolean();
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
}

void Solver::SetupGas() {
    std::cout << "Solver::SetupGas()" << std::endl;

    gas = newPhase(mech_file,mech_type);
    std::vector<ThermoPhase*> phases_ {gas};
    kin = newKineticsMgr(gas->xml(),phases_);
    trans = newDefaultTransportMgr(gas);

    M = m + gas->nSpecies();

    if (verbose) {
        gas->setState_TPX(T_in, p_sys, X_0);
        std::cout << "  SetupGas() at T = " << gas->temperature() << " and p = " << gas->pressure()
                  << " gives viscosity = " << trans->viscosity() << " for X = " << X_0 << std::endl;
    }
}

void Solver::DerivedParams() {
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
    trans->getMixDiffCoeffs(mix_diff_coeffs.data());

    // Molar production rates
    omega_dot_mol.resize(gas->nSpecies());
    kin->getNetProductionRates(omega_dot_mol.data());

    // Species molar enthalpies
    species_enthalpies_mol.resize((gas->nSpecies()));
    gas->getPartialMolarEnthalpies(species_enthalpies_mol.data());

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
                if (IC_type == "linear_T"){
                    phi.col(k) = VectorXd::LinSpaced(N,T_wall,T_in);
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
            // TODO ensure that this is mathematically correct for Z_l and m_d BCs
            // Z_l
            case 2:
                break;
            // m_d
            case 3:
                break;
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
            // Z_l
            case 2:
                if (evaporating)
                    phi(N-1,k) = Z_l_in;
                else
                    phi(N-1,k) = 0.0;
                break;
            // m_d
            case 3:
                if (evaporating)
                    phi(N-1,k) = m_d_in;
                else
                    phi(N-1,k) = 1.0e-300;
                break;
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

void Solver::Clipping(){
    // TODO this is a hack and is slow
    for (int i = 0; i < N; i++){
        for (int k = m; k < M; k++){
            if (phi(i,k) < 0.0)
                phi(i,k) = 0.0;
        }
    }
}

void Solver::StepIntegrator() {
    // TODO make polymorphic!!
    // Fwd Euler
    // Get RHS
    MatrixXd RHS = GetRHS(time,phi);
    // Don't update V, T, or Y_k (these have wall BCs, the spray advection eqns do not)
    RHS.row(0).head(2) = RowVectorXd::Zero(2);
    RHS.row(0).tail(gas->nSpecies()) = RowVectorXd::Zero(gas->nSpecies());
    // Don't update anything at inlet (we impose BCs for all variables here)
    RHS.row(RHS.rows()-1) = RowVectorXd::Zero(RHS.cols());

    phi = phi + dt*RHS;
}

double Solver::Getu(const Ref<const MatrixXd>& phi, int i){
    if (i == 0){
        // no-slip wall
        return 0.0;
    } else {
        //TODO this can be made vastly more efficient by keeping track of the integral
        VectorXd rho_vec = Getrho(phi.topRows(i + 1));
        VectorXd V_vec = phi.col(0).head(i + 1);
        return -(2.0 / rho_vec(i)) * Quadrature(rho_vec.array() * V_vec.array(), dx.head(i));
    }
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

double Solver::Getomegadot(const Ref<const RowVectorXd>& phi_, int k) {
    double omegadot_;
    switch (k){
        // V
        case 0:
            omegadot_ = rho_inf * pow(a, 2) - gas->density() * pow(phi_(0), 2);
            break;
        // T: - SUM_(i = 0)^(nSpecies) h_i^molar * omegadot_i^molar
        case 1:
            if (reacting)
                omegadot_ = - species_enthalpies_mol.dot(omega_dot_mol);
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
            if (reacting)
                omegadot_ = omega_dot_mol(k-m)*gas->molecularWeight(k-m);
            else
                omegadot_ = 0.0;
    }
    return omegadot_;
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
            double A_ = 6.0/M_PI * m_d_/rho_l;
            // energy analogy mass transfer Spalding number
            double B_ = gas->cp_mass()*(T_ - T_l)/L_v;
            mdot_ = -2.0*M_PI*rho_*pow(A_,1.0/3.0)*mix_diff_coeffs(fuel_idx)*log(1.0 + B_);
        } else {
            mdot_ = 0.0;
        }
        // TODO for Getmdot_liq, this is time stepping scheme dependent, and doesn't account for transport!!! Must think of a better way!!!
        if (m_d_ + dt*mdot_ < 0.0 || mdot_ > 0.0){
            //mdot_ = m_d_/dt + 1.0e-300;
            mdot_ = 0.0; // this is garbage, just a hack to get it running
        }
            return mdot_;
    } else {
        mdot_ = 0.0;
    }
    return mdot_;
}

MatrixXd Solver::GetRHS(double time, const Ref<const MatrixXd>& phi){

    // initialize vectors and matrices
    VectorXd u(VectorXd::Zero(N));
    VectorXd rho_inv(VectorXd::Zero(N));
    MatrixXd c(MatrixXd::Zero(N,M));
    MatrixXd mu(MatrixXd::Zero(N,M));
    MatrixXd omegadot(MatrixXd::Zero(N,M));
    VectorXd mdot_liq(VectorXd::Zero(N));
    MatrixXd Gammadot(MatrixXd::Zero(N,M));

    for (int i = 0; i < N; i++){
        u(i) = Getu(phi, i);
        SetState(phi.row(i));
        rho_inv(i) = 1.0/gas->density();
        mdot_liq(i) = Getmdot_liq(phi.row(i));
        for (int k = 0; k < M; k++){
            c(i, k) = Getc(k);
            mu(i, k) = Getmu(k);
            omegadot(i, k) = Getomegadot(phi.row(i),k);
            Gammadot(i,k) = GetGammadot(phi.row(i),k);
        }
    }

    /*
     * RHS          = conv + diff + src_gas + src_spray
     * conv         = -u*ddx*phi
     * diff         = (diag(rho_inv)*c) .* (ddx * (mu .* (ddx * phi))) (alternative)
     * diff         = (diag(rho_inv)*c) .* (mu .* (d2dx2 * phi))
     * src_gas      = (diag(rho_inv)*c) .* omegadot
     * src_spray    = (diag(rho_inv)*c) .* (diag(mdot_liq) * Gammadot) (pure fuel)
     */
    MatrixXd conv = -1.0*u.asDiagonal() * (ddx * phi);
    MatrixXd diff = (rho_inv.asDiagonal() * c).array() * (mu.array() * (d2dx2 * phi).array());
    MatrixXd src_gas  = (rho_inv.asDiagonal() * c).array() * omegadot.array();
    MatrixXd src_spray = (rho_inv.asDiagonal() * c).array() * (mdot_liq.asDiagonal() * Gammadot).array();

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
                if (verbose){
                    std::cout << "iteration = " << iteration << std::endl;
                    std::cout << "phi(t = " << time << ") = \n" << phi << std::endl;
                }
            }

            // Loop on BCs
            //inlet_bc.Update();
            //wall_bc.Update();
            UpdateBCs(); // non-polymorphic version for initial testing

            // Integrate ODE
            //integrator.Step();
            StepIntegrator();// non-polymorphic version for initial testing

            // TODO upgrade numerics so clipping isn't necessary for species
            Clipping();

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