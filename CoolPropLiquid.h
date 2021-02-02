//
// Created by Danyal Mohaddes on 1/28/21.
//

#ifndef HIEMENZ_SOLVER_COOLPROPLIQUID_H
#define HIEMENZ_SOLVER_COOLPROPLIQUID_H

#include "Liquid.h"

class CoolPropLiquid : public Liquid {
public:
  CoolPropLiquid(std::string X_liq_);
  ~CoolPropLiquid();
  double T_sat(double p) const;
  double p_sat(double T) const;
  double L_v(double T) const;
  double rho_liq(double T, double p) const;
  double rho_satvap(double T) const;
  double cp_liq(double T, double p) const;
  double cp_satvap(double T) const;
  double lambda_satvap(double T) const;
  double mu_satvap(double T) const;

private:
  std::string GetCoolPropName(const std::string cantera_name);
  std::string GetCoolPropString(std::string cantera_string);

  std::string coolprop_string;
};


#endif //HIEMENZ_SOLVER_COOLPROPLIQUID_H
