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



#endif //HIEMENZ_SOLVER_MESHING_H
