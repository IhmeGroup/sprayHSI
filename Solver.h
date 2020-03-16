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
    void ConstructMesh();
    void ConstructOperators();
    void SetIC();
    int RunSolver();

private:
    bool CheckStop();
    void Output();
    void StepIntegrator();
    void UpdateBCs();
    MatrixXd GetRHS(double time, const Ref<const MatrixXd>& phi);
    double Getu(const Ref<const MatrixXd>& phi, int i);
    double Quadrature(const Ref<const VectorXd>& rhoV_, const Ref<const VectorXd>& dx_);
    VectorXd Getrho(const Ref<const MatrixXd>& phi);
    void SetState(const Ref<const RowVectorXd>& phi);
    double Getc(int k);
    double Getmu(int k);
    double Getomegadot(const Ref<const RowVectorXd>& phi, int k);

    bool verbose;

    std::string input_file;

    int N;
    int m;
    int M;
    double L;

    VectorXd nodes;
    VectorXd dx;

    MatrixXd ddx;
    MatrixXd d2dx2;

    std::string mech_file;
    std::string mech_type;
    ThermoPhase* gas;
    Kinetics* kin;
    Transport* trans;

    double Tgas_0;

    double T_in;
    double T_wall;
    double p_sys;
    double mdot;
    double a;

    double rho_inf;

    /*
     * Solution tensor \\phi, NxM
     * \\(.) is a tensor, \(.) is a vector
     * \\phi = [\V, \T, \Y1, ..., \Ym, \Zl, \md]
     */
    MatrixXd phi;

    int iteration;
    int output_interval; //TODO could add an output interval in time as well
    double time;
    double dt;

    double time_max;
    double iteration_max;
};


#endif //HIEMENZ_SOLVER_SOLVER_H
