//
// Created by Danyal Mohaddes on 2/1/21.
//

#ifndef HIEMENZ_SOLVER_FITLIQUID_H
#define HIEMENZ_SOLVER_FITLIQUID_H

#include "Liquid.h"

/*
 * Liquid property class using algebraic fits to property data
 */
class FitLiquid : public Liquid {

public:
  FitLiquid(std::string X_liq_);
  ~FitLiquid();
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

  double MW;
  double Tcrit, rho_c;
  double ST0, ST1;
  double DV0, DV1;
  double DL0, DL1, DL2;
  double PV0, PV1, PV2;
  double HV0, HV1, HV2;
  double VV0, VV1, VV2, VV3;
  double KV0, KV1, KV2, KV3, KV4;
  double CV0, CV1, CV2, CV3, CV4;
  double CL0, CL1, CL2, CL3, CL4;
};


#endif //HIEMENZ_SOLVER_FITLIQUID_H
