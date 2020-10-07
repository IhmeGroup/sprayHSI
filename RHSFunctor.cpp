//
//  Created by Danyal Mohaddes on 9/9/20.
//

#include <iostream>
#include "RHSFunctor.h"
#include "Eigen/Dense"

RHSFunctor::RHSFunctor(int n, int m, Solver* my_p_solver) : N(n), M(m), p_solver(my_p_solver) {}
RHSFunctor::~RHSFunctor(){
  std::cout << "RHSFunctor::~RHSFunctor()" << std::endl;
}

int RHSFunctor::rhs(double t, double *ydata, double *ydotdata) {
  try {
    Eigen::MatrixXd phi_ = Eigen::Map<Eigen::MatrixXd>(ydata, N, M);
    Eigen::Map<Eigen::MatrixXd>(ydotdata, N, M) = p_solver->GetRHS(t, phi_);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << "cvode_rhs: unhandled exception" << std::endl;
    return -1; // unrecoverable error
  }
  return 0;
}