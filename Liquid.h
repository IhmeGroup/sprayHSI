//
// Created by Danyal Mohaddes on 1/28/21.
//

#ifndef HIEMENZ_SOLVER_LIQUID_H
#define HIEMENZ_SOLVER_LIQUID_H


#include <string>

class Liquid {

public:
  Liquid(std::string X_liq_) : X_liq(X_liq_) {}

  virtual double T_sat(double p) const = 0;
  virtual double p_sat(double T) const = 0;
  virtual double L_v(double p) const = 0;
  virtual double rho_liq(double T, double p) const = 0;
  virtual double rho_satvap(double p) const = 0;
  virtual double cp_liq(double T, double p) const = 0;
  virtual double cp_satvap(double p) const = 0;
  virtual double lambda_satvap(double p) const = 0;
  virtual double mu_satvap(double p) const = 0;

protected:
  const std::string X_liq;
};


#endif //HIEMENZ_SOLVER_LIQUID_H
