//
//  Created by Danyal Mohaddes on 9/9/20.
//

#ifndef HIEMENZ_SOLVER_RHSFUNCTOR_H
#define HIEMENZ_SOLVER_RHSFUNCTOR_H

#include "Solver.h"

class RHSFunctor {

private:
  int N;
  int M;
  Solver* p_solver;


public:
  RHSFunctor(int n, int m, Solver* my_p_solver);
  ~RHSFunctor();

  int rhs(double t, double* ydata, double* ydotdata);
};


#endif //HIEMENZ_SOLVER_RHSFUNCTOR_H
