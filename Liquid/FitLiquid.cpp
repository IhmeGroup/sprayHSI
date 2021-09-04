// This file is part of the 1D-HSI solver hosted at github.com/IhmeGroup/sprayHSI
// D. Mohaddes
// September 2021

//
// Created by Danyal Mohaddes on 2/1/21.
//

#include "FitLiquid.h"
#include <iostream>
#include <cmath>

#define NEAR_ONE 0.9999999

// currently fit to n-dodecane at 1 atm

FitLiquid::FitLiquid(std::string X_liq_) : Liquid(X_liq_) {
  if (X_liq == "NC12H26:1.0"){
    // Values taken from CharlesX, CtiLiquid.h for NISTLiquid: n-dodecane

    // saturated liquid/vapor data from NIST (REFPROP)
    // valid for 265K < T < Tcrit

    MW = 170.3348; // molecular weight [kg/kmol]

    // critical properties
    Tcrit = 658.1;
    rho_c = 226.55;

    // vapor pressure //not the same as WebBook's antoine coeffs...
    PV0 = 9.32406;
    PV1 = 1783.72;
    PV2 = -76.8171;

    // heat of evaporation
    HV0 = 503310;
    HV1 = -0.236108;
    HV2 = 0.371665;

    // surface tension
    ST0 = 0.0529379;
    ST1 = 1.32243;

    // liquid density
    DL0 = 737.626;
    DL1 = -0.30859;
    DL2 = 0.347159;

    // liquid heat capacity
    CL0 = 7.56655;
    CL1 = 0.211997;
    CL2 = -5.52206;
    CL3 = 3.36639;
    CL4 = 0.362292;

    // vapor density
    DV0 = 1.07723;
    DV1 = 0.397863;

    // vapor heat capacity
    CV0 = 6.69259;
    CV1 = 0.681822;
    CV2 = -2.47209;
    CV3 = 1.2723;
    CV4 = 0.248697;

    // vapor conductivity
    KV0 = -6.96452;
    KV1 = 3.55663;
    KV2 = -0.619386;
    KV3 = 0.911432;
    KV4 = 0.0375757;

    // vapor viscosity
    VV0 = 1.05194e-05;
    VV1 = -3.85048e-07;
    VV2 = 8.19302e-06;
    VV3 = 0.0035629;
  } else {
    std::cerr << "Unrecognized liquid composition " << X_liq << std::endl;
    throw(0);
  }
}

double FitLiquid::T_sat(double p) const {
  // inverted Antoine equation [K]
  return PV1 / (PV0 - log10(p)) - PV2;
}

double FitLiquid::p_sat(double T) const {
  // Antoine equation [Pa]
  return (pow(10.0, PV0 - PV1 / (T + PV2)));
}

double FitLiquid::L_v(double T) const {
  // heat of evaporation [J/kg]
  double t = std::min(1.0, T / Tcrit);
  return (HV0 * exp(HV1 * t) * pow(1.0 - t, HV2));
  // NOTE: dHv = 0 when T/Tcrit = 1
}

double FitLiquid::rho_liq(double T, double p) const {
  // liquid density [kg/m3], assumed independent of pressure
  double t = std::min(1.0, T / Tcrit);
  return (DL0 * exp(DL1 * t) * pow(1.0 - t, DL2) + rho_c);
  // NOTE: rho = rho_c when T/Tcrit = 1
}

double FitLiquid::rho_vap(double T, double p) const {
  // assume ideal gas law
  // this is how CharlesX calculates rho_satvap, with p = p_sys
  return p / (8314.0/MW * T);
};

double FitLiquid::rho_satvap(double T) const {
  // vapour density from the ideal gas law, assuming a pure vapour on the gas side of the interface at temperature T
  // take pressure as p_sat
  return p_sat(T)/ (8314.0/MW * T);
}

double FitLiquid::cp_liq(double T, double p) const {
  // liquid heat capacity [J/kg/K]
  double t = std::min(NEAR_ONE, T / Tcrit);
  return (exp(CL0 + CL1 * pow(t, CL2 * t + CL3) / pow(1.0 - t, CL4)));
  // NOTE: cp = Inf when T/Tcrit = 1
}

double FitLiquid::cp_satvap(double T) const {
  // vapor heat capacity [J/kg/K]
  double t = std::min(NEAR_ONE, T / Tcrit);
  return (exp(CV0 + CV1 * pow(t, CV2 * t + CV3) / pow(1.0 - t, CV4)));
  // NOTE: cpv = Inf when T/Tcrit = 1
}

double FitLiquid::lambda_satvap(double T) const {
  // vapor conductivity [W/m/K]
  double t = std::min(NEAR_ONE, T / Tcrit);
  return (exp(KV0 + KV1 * pow(t, KV2 * t + KV3) / pow(1.0 - t, KV4)));
  // NOTE: kv = Inf when T/Tcrit = 1
}

double FitLiquid::mu_satvap(double T, double p) const {
  // vapor viscosity [kg/m/s]
  double t = std::min(1.0, T / Tcrit);
  double w = 0.5 * (1.0 + tanh(40.0 * (t - 0.9)));
  double mu0 = VV0 * t + VV1;
//  double mu1 = VV2 * exp(VV3 * rho_satvap(T));
  double mu1 = VV2 * exp(VV3 * rho_vap(T, p));
  return ((1.0 - w) * mu0 + w * mu1);
}

double FitLiquid::D_satvap(double T, double p) const {
  // Vapour diffusivity [m^2/s]
  // assume Le = 1
//  return lambda_satvap(T) / (rho_satvap(T) * cp_satvap(T));
  return lambda_satvap(T) / (rho_vap(T, p) * cp_satvap(T));
}

FitLiquid::~FitLiquid(){
  std::cout << "FitLiquid::~FitLiquid()" << std::endl;
}
