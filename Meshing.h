// This file is part of the 1D-HSI solver hosted at github.com/IhmeGroup/sprayHSI
// D. Mohaddes
// September 2021

//
// Created by Danyal Mohaddes on 10/20/20.
//

#ifndef HIEMENZ_SOLVER_MESHING_H
#define HIEMENZ_SOLVER_MESHING_H

#include <boost/math/tools/roots.hpp>

template <class T>
struct GeometricRatioFunctor
{
  //  root finding problem, no derivatives.
  GeometricRatioFunctor(T const& N_, T const& L_, T const& D0_) : N(N_), L(L_), D0(D0_)
  { /* Constructor just stores value a to find root of. */ }
  T operator()(T const& r)
  {
    T A_ = L/D0;
    T fx = pow(r, N) - A_*r + A_ - 1.0; // root finding problem
    return fx;
  }
private:
  T N;
  T L;
  T D0;
};

template <class T>
struct TWallFunctor
{
  //  root finding problem, no derivatives.
  TWallFunctor(ThermoPhase* gas_, Transport* trans_, T const& T_in_, T const& T_ext_, T const& p_sys_,
          std::string const& X_0_, T const& L_, T const& L_s_, T const& lam_s_) : gas(gas_), trans(trans_), T_in(T_in_), T_ext(T_ext_),
                                                      p_sys(p_sys_), X_0(X_0_), L(L_), L_s(L_s_), lam_s(lam_s_) {}
  T operator()(T const& T_w)
  {
    T Tbar = (T_in + T_w)/2.0;
    gas->setState_TPX(Tbar, p_sys, X_0);
    T lam_g = trans->thermalConductivity();

    /*
     * Approximate steady-state conjugate HT problem, to be used as initial condition for IC_type = "linear_T"
     * lambda_g(T_bar) / L * (T_wall - T_inlet) = lambda_s / L_s * (T_s_ext - T_wall),
     * where T_bar = (T_inlet + T_wall)/2
     */

    T fx = (lam_g/L) * (T_w - T_in) - (lam_s/L_s) * (T_ext - T_w); // root finding problem
    return fx;
  }
private:
  ThermoPhase* gas;
  Transport* trans;
  T T_in;
  T T_ext;
  T p_sys;
  std::string X_0;
  T L;
  T L_s;
  T lam_s;
};

template <class T>
T GetGeometricRatio(int N,double L, double D0){
  // From https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/root_finding_examples/cbrt_eg.html
  // return cube root of x using bracket_and_solve (no derivatives).
  using namespace std;                          // Help ADL of std functions.
  using namespace boost::math::tools;           // For bracket_and_solve_root.

  T guess = 1.0 + 1.0e-3;                                // Rough guess
  T factor = 2;                                 // How big steps to take when searching.

  const boost::uintmax_t maxit = 100;           // Limit to maximum iterations.
  boost::uintmax_t it = maxit;                  // Initally our chosen max iterations, but updated with actual.
  bool is_rising = true;                        // So if result if guess^3 is too low, then try increasing guess.
  int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  // Some fraction of digits is used to control how accurate to try to make the result.
  int get_digits = digits - 3;                  // We have to have a non-zero interval at each step, so
  // maximum accuracy is digits - 1.  But we also have to
  // allow for inaccuracy in f(x), otherwise the last few
  // iterations just thrash around.
  eps_tolerance<T> tol(get_digits);             // Set the tolerance.
  std::pair<T, T> r = bracket_and_solve_root(GeometricRatioFunctor<T>(N, L, D0), guess, factor, is_rising, tol, it);
  return r.first + (r.second - r.first)/2;      // Midway between brackets is our result, if necessary we could
  // return the result as an interval here.
}

template <class T>
T GetTWall(ThermoPhase* gas, Transport* trans, T T_in, T T_ext, T p_sys,
           std::string X_0, T L, T L_s, T lam_s){
  // From https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/root_finding_examples/cbrt_eg.html
  // return cube root of x using bracket_and_solve (no derivatives).
  using namespace std;                          // Help ADL of std functions.
  using namespace boost::math::tools;           // For bracket_and_solve_root.

  T guess = T_ext;                                // Rough guess
  T factor = 2;                                 // How big steps to take when searching.

  const boost::uintmax_t maxit = 100;           // Limit to maximum iterations.
  boost::uintmax_t it = maxit;                  // Initally our chosen max iterations, but updated with actual.
  bool is_rising = true;                        // So if result if guess^3 is too low, then try increasing guess.
  int digits = std::numeric_limits<T>::digits;  // Maximum possible binary digits accuracy for type T.
  // Some fraction of digits is used to control how accurate to try to make the result.
  int get_digits = digits - 3;                  // We have to have a non-zero interval at each step, so
  // maximum accuracy is digits - 1.  But we also have to
  // allow for inaccuracy in f(x), otherwise the last few
  // iterations just thrash around.
  eps_tolerance<T> tol(get_digits);             // Set the tolerance.
  std::pair<T, T> r = bracket_and_solve_root(TWallFunctor<T>(gas,trans,T_in,T_ext,p_sys,X_0,L,L_s,lam_s), guess, factor, is_rising, tol, it);
  return r.first + (r.second - r.first)/2;      // Midway between brackets is our result, if necessary we could
  // return the result as an interval here.
}

#endif //HIEMENZ_SOLVER_MESHING_H
