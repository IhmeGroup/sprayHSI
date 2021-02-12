//
// Created by Danyal Mohaddes on 2/4/21.
//

#include "Solver.h"

#define NEAR_ONE 0.9999999999

double Solver::Getu(const Ref<const MatrixXd>& Phi_, int i){
  if (i == 0){
    // no-slip wall
    return 0.0;
  } else {
    // this could be made somewhat more efficient by keeping track of the integral
    VectorXd rho_vec = Getrho(Phi_.topRows(i + 1));
    VectorXd V_vec = Phi_.col(idx_V).head(i + 1);
    return -(2.0 / rho_vec(i)) * Quadrature(rho_vec.array() * V_vec.array(), dx.head(i));
  }
}

double Solver::GetZBilger(const Ref<const MatrixXd>& Phi_, int i){
  // This function is mostly from CharlesX, PhysicsDeriveFunctor.h, Hao Wu 2017.

  /*
   *             2(Y_C - Yo_C)/W_C + (Y_H - Yo_H)/2W_H + (Y_O - Yo_O)/W_O
   * ZBilger =  -----------------------------------------------------------
   *            2(Yf_C - Yo_C)/W_C + (Yf_H - Yo_H)/2W_H + (Yf_O - Yo_O)/W_O
   */

  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();

  size_t i_C = gas->elementIndex("C");
  size_t i_H = gas->elementIndex("H");
  size_t i_O = gas->elementIndex("O");

  double W_C = gas->atomicWeight(i_C);
  double W_H = gas->atomicWeight(i_H);
  double W_O = gas->atomicWeight(i_O);

  gas->setMoleFractionsByName(X_ox);
  double Yo_C = gas->elementalMassFraction(i_C);
  double Yo_H = gas->elementalMassFraction(i_H);
  double Yo_O = gas->elementalMassFraction(i_O);

  gas->setMoleFractionsByName(X_f);
  double Yf_C = gas->elementalMassFraction(i_C);
  double Yf_H = gas->elementalMassFraction(i_H);
  double Yf_O = gas->elementalMassFraction(i_O);

  double denom_ = 2.0 * (Yf_C - Yo_C) / W_C
                  + 0.5 * (Yf_H - Yo_H) / W_H
                  + (Yf_O - Yo_O) / W_O;

  SetState(Phi_.row(i));
  double num_   = 2.0 * (gas->elementalMassFraction(i_C) - Yo_C) / W_C
                  + 0.5 * (gas->elementalMassFraction(i_H) - Yo_H) / W_H
                  + (gas->elementalMassFraction(i_O) - Yo_O) / W_O;
  return num_/denom_;
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

VectorXd Solver::Getrho(const Ref<const MatrixXd>& phi_){
  VectorXd rho_vec(VectorXd::Zero(phi_.rows()));
  int thread = omp_get_thread_num();
  for (int i = 0; i < phi_.rows(); i++){
    SetState(phi_.row(i));
    rho_vec(i) = gas_vec[thread]->density();
  }
  return rho_vec;
}

void Solver::SetState(const Ref<const RowVectorXd>& phi_){
  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();
  gas->setState_TPY(phi_(idx_T),p_sys,phi_.tail(gas->nSpecies()).data());
}

void Solver::SetGasQWall() {
  int thread = omp_get_thread_num();
  Transport* trans_ = trans_vec[thread].get();
  SetState(phi.row(0));
  double lam_g_ = trans_->thermalConductivity();
  double dTgdx_w = (phi(0, idx_T) - wall_interior_BC(idx_T))/dx(0);
  q_wall = lam_g_ * dTgdx_w;

  if (verbose)
    std::cout << "q_wall = " << q_wall << "W/m2" << std::endl;

}

double Solver::Getc(const int k) {
  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();

  double c_;
  switch (k){
    // T
    case idx_T:
      c_ = 1.0/gas->cp_mass();
      break;
    default:
      c_ = 1.0;
  }
  return c_;
}

double Solver::Getmu(const int k) {
  int thread = omp_get_thread_num();
  Transport* trans = trans_vec[thread].get();

  double mu_;
  switch (k){
    // V
    case idx_V:
      mu_ = trans->viscosity();
      break;
    // T
    case idx_T:
      mu_ = trans->thermalConductivity();
      break;
    // Z_l
    case idx_Z_l:
      mu_ = 0.0;
      break;
    // m_d
    case idx_m_d:
      mu_ = 0.0;
      break;
    // T_d
    case idx_T_d:
      mu_ = 0.0;
      break;
    // Species
    default:
      mu_ = mix_diff_coeffs_vec[thread](k - m);
  }
  return mu_;
}

double Solver::Getmu_av(const int k) {
  // Artificial viscosity
  double mu_av_;
  switch (k){
    // Z_l
    case idx_Z_l:
      mu_av_ = av_Zl; // TODO make this physics-based
      break;
    // m_d
    case idx_m_d:
      mu_av_ = av_md; // TODO make this physics-based
      break;
    // T_d
    case idx_T_d:
      mu_av_ = av_Td; // TODO make this physics-based
      break;
    // Other quantities receive no artificial viscosity
    default:
      mu_av_ = 0.0;
  }
  return mu_av_;
}

double Solver::GetDd(const double m_d_, const double T_d_) {
  if (m_d_ > 0.0)
    return pow(m_d_/(M_PI / 6.0 * liq->rho_liq(T_d_, p_sys)), 1.0/3.0);
  else
    return 0.0;
}

double Solver::GetNu(const Ref<const RowVectorXd>& phi_) {
  return 2.0; // TODO upgrade this when slip velocity added
}

double Solver::GetSh(const Ref<const RowVectorXd>& phi_) {
  return 2.0; // TODO upgrade this when slip velocity added
}

double Solver::GetBeta(const Ref<const RowVectorXd>& phi_, const double mdot_liq_) {
  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();
  Transport* trans = trans_vec[thread].get();

  double T_d_ = std::min(NEAR_ONE*T_l, phi_(idx_T_d));
  double Y_g_ = std::min(NEAR_ONE, phi_(fuel_idx + m));
  double m_d_ = phi_(idx_m_d);
  double D_d_ = GetDd(m_d_, T_d_);

  double M_m = gas->meanMolecularWeight();
  double M_f = gas->molecularWeight(fuel_idx);
  double theta_2 = M_m/M_f;
  double chi_seq = std::min(NEAR_ONE, liq->p_sat(T_d_)/p_sys);
  double Y_seq = chi_seq/(chi_seq + (1.0 - chi_seq)*theta_2);
  // reference mass fraction (1/3 rule)
  double Yref_ = (1.0-A_ref) * Y_seq + A_ref * Y_g_;
  // reference properties
  double cp_ = Yref_ * liq->cp_satvap(T_d_) + (1.0 - Yref_) * gas->cp_mass();
//  double rho_ = Yref_ * liq->rho_satvap(T_d_) + (1.0 - Yref_) * gas->density();
  double rho_ = Yref_ * liq->rho_vap(T_d_, p_sys) + (1.0 - Yref_) * gas->density();
  double lambda_ = Yref_ * liq->lambda_satvap(T_d_) + (1.0 - Yref_) * trans->thermalConductivity();
  double mu_ = Yref_ * liq->mu_satvap(T_d_, p_sys) + (1.0 - Yref_) * trans->viscosity();

  double beta = -((liq->rho_liq(T_d_, p_sys) * cp_ * pow(D_d_, 2))/(12.0 * lambda_)) * (mdot_liq_ / m_d_); // previous time step's mdot_liq
  return beta;
}

double Solver::Getf2(const Ref<const RowVectorXd>& phi_, const double mdot_liq_){
  // Model M7, Miller et al. 1998
  double beta = GetBeta(phi_, mdot_liq_);
  double f2 = (abs(beta) < 1e-12) ? 1.0 : beta / (exp(beta) - 1.0);
  return f2;
}

double Solver::Getomegadot(const Ref<const RowVectorXd>& phi_, const int k, const int idx) {
  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();
  Transport* trans = trans_vec[thread].get();

  double omegadot_ = 0.0;
  double rho_ = gas->density();
  double V_ = phi_(idx_V);
  double T_ = phi_(idx_T);
  double Z_l_ = phi_(idx_Z_l);
  double m_d_ = phi_(idx_m_d);
  double T_d_ = phi_(idx_T_d);
  double Y_g_ = std::min(NEAR_ONE, phi_(fuel_idx + m));
  double D_d_;
  double lambda_;
  if (spray) {
    D_d_ = GetDd(m_d_, T_d_);
    double M_m = gas->meanMolecularWeight();
    double M_f = gas->molecularWeight(fuel_idx);
    double theta_2 = M_m / M_f;
    double chi_seq = std::min(NEAR_ONE, liq->p_sat(T_d_) / p_sys);
    double Y_seq = chi_seq / (chi_seq + (1.0 - chi_seq) * theta_2);
    // reference mass fraction (1/3 rule)
    double Yref_ = (1.0 - A_ref) * Y_seq + A_ref * Y_g_;
    // reference properties
    lambda_ = Yref_ * liq->lambda_satvap(T_d_) + (1.0 - Yref_) * trans->thermalConductivity();
  }

  switch (k){
    // V: rho_inf * a^2 - rho * V^2
    case idx_V:
      omegadot_ = rho_inf * pow(a, 2) - rho_ * pow(V_, 2);
      break;

    // T:
    case idx_T:
      // spray: - (rho*Z_l/m_d) * m_d * c_l * f2 * (6 Nu * lamba) / (c_l * rho_l * D_d^2) * (T - T_d)
      if (spray) {
        if (D_d_ > D_min && T_d_ < T_l) {
          omegadot_ += -rho_ * Z_l_ * Getf2(phi_, mdot_liq(idx)) * (6.0 * GetNu(phi_) * lambda_) /
                       (liq->rho_liq(T_d_, p_sys) * pow(D_d_, 2)) * (T_ - T_d_);
        }
      }
      // rxn: - SUM_(i = 0)^(nSpecies) h_i^molar * omegadot_i^molar,
      if (reacting) {
        omegadot_ += -species_enthalpies_mol_vec[thread].dot(omega_dot_mol_vec[thread]);
      }
      break;

    // Z_l: 0
    case idx_Z_l:
      omegadot_ = 0.0;
      break;

    // m_d: 0
    case idx_m_d:
      omegadot_ = 0.0;
      break;

    // T_d: + rho * f2 * (Nu/(3Pr)) * (theta_1/tau_d) * (T - T_d) = rho * f2 * (6 Nu * lamba) / (c_l * rho_l * D_d^2) * (T - T_d)
    case idx_T_d:
      omegadot_ += Tdot_liq_1(idx);
      break;

    // Species: omegadot_i^molar * molarmass_i
    default:
      if (reacting) {
        omegadot_ += omega_dot_mol_vec[thread](k - m) * gas->molecularWeight(k - m);
      }
  }
  return omegadot_;
}

double Solver::GetGammadot(const Ref<const RowVectorXd>& phi_, const int k, const int idx){
  if (!spray) return 0.0;

  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();

  double gammadot_;
  double rho_ = gas->density();
  double T_ = phi_(idx_T);
  double Z_l_ = phi_(idx_Z_l);
  double m_d_ = phi_(idx_m_d);
  double T_d_ = phi_(idx_T_d);
  switch (k){
    // V
    case idx_V:
      gammadot_ = 0.0;
      break;

    // T: -(rho*Z_l/m_d) * (-1) * (cp * (T - T_d) + L_v)
    case idx_T:
      gammadot_ = - (rho_ * Z_l_ / m_d_) * -1.0 * (gas->cp_mass() * (T_ - T_d_) + L_v); // TODO should be vapour c_p
      break;

    // Z_l: + (rho*Z_l/m_d)
    case idx_Z_l:
      gammadot_ = rho_ * Z_l_ / m_d_;
      break;

    // m_d: + rho
    case idx_m_d:
      gammadot_ = rho_;
      break;

    // T_d: + (rho * L_v) / (c_l * m_d)
    case idx_T_d:
      gammadot_ = Tdot_liq_2(idx);
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

double Solver::GetHM(const Ref<const RowVectorXd>& phi_, const double mdot_liq_) {
  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();
  Transport* trans = trans_vec[thread].get();

  double T_d_ = std::min(NEAR_ONE*T_l, phi_(idx_T_d));
  double Y_g_ = std::min(NEAR_ONE, phi_(fuel_idx + m));
  double m_d_ = phi_(idx_m_d);
  double D_d_ = GetDd(m_d_, T_d_);
  double M_m = gas->meanMolecularWeight();
  double M_f = gas->molecularWeight(fuel_idx);
  double theta_2 = M_m/M_f;
  double chi_seq = std::min(NEAR_ONE, liq->p_sat(T_d_)/p_sys);
  double Y_seq = chi_seq/(chi_seq + (1.0 - chi_seq)*theta_2);
  // reference mass fraction (1/3 rule)
  double Yref_ = (1.0-A_ref) * Y_seq + A_ref * Y_g_;
  // reference properties
  double cp_ = Yref_ * liq->cp_satvap(T_d_) + (1.0 - Yref_) * gas->cp_mass();
//  double rho_ = Yref_ * liq->rho_satvap(T_d_) + (1.0 - Yref_) * gas->density();
  double rho_ = Yref_ * liq->rho_vap(T_d_, p_sys) + (1.0 - Yref_) * gas->density();
  double lambda_ = Yref_ * liq->lambda_satvap(T_d_) + (1.0 - Yref_) * trans->thermalConductivity();
  double mu_ = Yref_ * liq->mu_satvap(T_d_, p_sys) + (1.0 - Yref_) * trans->viscosity();
  // Miller et al 1998, model M7
  double Sc = mu_/(rho_ * liq->D_satvap(T_d_, p_sys));
  double L_k = (mu_ * pow(2.0 * M_PI * T_d_ * 8314.0/M_f ,0.5)) / (1.0 * Sc * p_sys);
  // use previous time step's mdot_liq, as suggested by Miller
  double beta = GetBeta(phi_, mdot_liq_);
  double chi_sneq = chi_seq - (L_k/(D_d_/2.0)) * beta;
  double Y_sneq = std::min(NEAR_ONE, chi_sneq/(chi_sneq + (1.0 - chi_sneq)*theta_2));
  double B_Mneq = (Y_sneq - Y_g_)/(1.0 - Y_sneq);
  return log(1.0 + B_Mneq);
}

double Solver::Getmdot_liq(const Ref<const RowVectorXd>& phi_, const double mdot_liq_){
  int thread = omp_get_thread_num();
  ThermoPhase* gas = gas_vec[thread].get();
  Transport* trans = trans_vec[thread].get();

  double mdot_;
  if (spray){
    double T_ = phi_(idx_T);
    double Z_l_ = phi_(idx_Z_l);
    double m_d_ = phi_(idx_m_d);
    double T_d_ = std::min(NEAR_ONE*T_l, phi_(idx_T_d));
    double Y_g_ = std::min(NEAR_ONE, phi_(fuel_idx + m));
    double D_d_ = GetDd(m_d_, T_d_);
    double M_m = gas->meanMolecularWeight();
    double M_f = gas->molecularWeight(fuel_idx);
    double theta_2 = M_m/M_f;
    double chi_seq = std::min(NEAR_ONE, liq->p_sat(T_d_)/p_sys);
    double Y_seq = chi_seq/(chi_seq + (1.0 - chi_seq)*theta_2);
    // reference mass fraction (1/3 rule)
    double Yref_ = (1.0-A_ref) * Y_seq + A_ref * Y_g_;
    // reference properties
    double cp_ = Yref_ * liq->cp_satvap(T_d_) + (1.0 - Yref_) * gas->cp_mass();
//      double rho_ = Yref_ * liq->rho_satvap(T_d_) + (1.0 - Yref_) * gas->density();
    double rho_ = Yref_ * liq->rho_vap(T_d_, p_sys) + (1.0 - Yref_) * gas->density();
    double lambda_ = Yref_ * liq->lambda_satvap(T_d_) + (1.0 - Yref_) * trans->thermalConductivity();
    double mu_ = Yref_ * liq->mu_satvap(T_d_, p_sys) + (1.0 - Yref_) * trans->viscosity();
    // TODO single component fuel assumed!!!
    if (m_d_ > 0.0 && Z_l_ > 0.0 && D_d_ > D_min){
      // Miller et al 1998, Model M7
      double Sh = GetSh(phi_);
      double Sc = mu_/(rho_ * liq->D_satvap(T_d_, p_sys));
      double tau_d = liq->rho_liq(T_d_, p_sys) * pow(D_d_, 2)/(18.0 * mu_);
      mdot_ = - Sh/(3.0 * Sc) * (m_d_/tau_d) * GetHM(phi_, mdot_liq_);
    } else {
      mdot_ = 0.0;
    }
    // Guard against condensation
    if (mdot_ > 0.0){
      mdot_ = 0.0;
    }
  } else {
    mdot_ = 0.0;
  }
  return mdot_;
}

MatrixXd Solver::GetRHS(double time_, const Ref<const MatrixXd>& phi_){
  // Loop on BCs
  SetBCs();

  // Create Phi = [wall_interior_BC, phi, inlet_BC]^T
  Phi << wall_interior_BC, phi_, inlet_BC;

  if (verbose) {
    std::cout << "GetRHS(t = " << time_ << ", phi)" << std::endl;
    std::cout << "  Phi = " << std::endl << Phi << std::endl;
  }

#pragma omp parallel for schedule(static,1) default(none)
  for (int i = 0; i < N; i++){

    int thread = omp_get_thread_num();
    ThermoPhase* gas = gas_vec[thread].get();

    u(i) = Getu(Phi, i+1);
    SetState(Phi.row(i+1));
    SetDerivedVars();
    rho_inv(i) = 1.0/gas->density();
    for (int k = 0; k < M; k++){
      c(i, k) = Getc(k);
      mu(i, k) = Getmu(k);
      mu_av(i, k) = Getmu_av(k);
      omegadot(i, k) = Getomegadot(Phi.row(i+1),  k, i);
      Gammadot(i,k) = GetGammadot(Phi.row(i+1), k, i);
    }
  }

  // TODO make AV smarter to only activate on strong gradients

  /*
   * RHS          = conv + diff + src_gas + src_spray                              (residual definition)
   * conv         = -u*ddx*Phi                                                     (convection)
   * diff         = (diag(rho_inv)*c) .* ((mu + mu_av) .* (d2dx2 * Phi))           (diffusion, as implemented)
   * diff         = (diag(rho_inv)*c) .* (ddx * (mu .* (ddx * Phi)))               (diffusion, alternative)
   * src_gas      = (diag(rho_inv)*c) .* omegadot                                  (gas source)
   * src_spray    = (diag(rho_inv)*c) .* (diag(mdot_liq) * Gammadot)               (spray source, pure fuels only)
   */
  conv = -1.0*u.asDiagonal() * (ddx * Phi);
  diff = (rho_inv.asDiagonal() * c).array() * ((mu + mu_av).array() * (d2dx2 * Phi).array());
  src_gas  = (rho_inv.asDiagonal() * c).array() * omegadot.array();
  src_spray = (rho_inv.asDiagonal() * c).array() * (mdot_liq.asDiagonal() * Gammadot).array();

  if (verbose)
    std::cout << "RHS = " << std::endl << conv + diff + src_gas + src_spray << std::endl;

  return conv + diff + src_gas + src_spray;
}

VectorXd Solver::GetSolidRHS(double time_, const Ref<const VectorXd>& T_s_) {
  // Set Solid BC from q_wall
  double T_w_ = T_s_(0) + q_wall * dx_s(0) / lam_s; // 1st order one-sided difference

  // Construct solution vector with BCs
  VectorXd T_s_vec_(2 + T_s_.size());
  T_s_vec_ << T_w_, T_s_, T_s_ext;

  if (verbose) {
    std::cout << "GetSolidRHS(t = " << time_ << ", T_s_)" << std::endl;
    std::cout << "  T_s_vec_ = " << std::endl << T_s_vec_ << std::endl;
  }

  // Get solid RHS
  VectorXd RHS_ = (lam_s/(rho_s * c_s)) * d2dx2_s * T_s_vec_;

  if (verbose)
    std::cout << "SolidRHS = " << std::endl << RHS_ << std::endl; // here, first cell RHS is ~-2e5.

  return RHS_;
}

void Solver::SetSprayRHS(){
  #pragma omp parallel for schedule(static,1) default(none)
  for (int i = 0; i < N; i++) {
    RowVectorXd phi_ = phi.row(i);

    SetState(phi_);

    double mdot_liq_ = Getmdot_liq(phi_, mdot_liq(i));
    int thread = omp_get_thread_num();
    ThermoPhase *gas = gas_vec[thread].get();
    Transport *trans = trans_vec[thread].get();

    double rho_ = gas->density();
    double T_ = phi_(idx_T);
    double Z_l_ = phi_(idx_Z_l);
    double m_d_ = phi_(idx_m_d);
    double T_d_ = phi_(idx_T_d);
    double Y_g_ = std::min(NEAR_ONE, phi_(fuel_idx + m));

    double D_d_;
    double lambda_;

    D_d_ = GetDd(m_d_, T_d_);
    double M_m = gas->meanMolecularWeight();
    double M_f = gas->molecularWeight(fuel_idx);
    double theta_2 = M_m / M_f;
    double chi_seq = std::min(NEAR_ONE, liq->p_sat(T_d_) / p_sys);
    double Y_seq = chi_seq / (chi_seq + (1.0 - chi_seq) * theta_2);
    // reference mass fraction (1/3 rule)
    double Yref_ = (1.0 - A_ref) * Y_seq + A_ref * Y_g_;
    // reference properties
    lambda_ = Yref_ * liq->lambda_satvap(T_d_) + (1.0 - Yref_) * trans->thermalConductivity();

    if (D_d_ > D_min && T_d_ < T_l) {
      Tdot_liq_1(i) = rho_ * Getf2(phi_, mdot_liq_) * (6.0 * GetNu(phi_) * lambda_) /
                      (liq->cp_liq(T_d_, p_sys) * liq->rho_liq(T_d_, p_sys) * pow(D_d_, 2)) * (T_ - T_d_);
    } else {
      Tdot_liq_1(i) = 0.0;
    }

    Tdot_liq_2(i) = (rho_ * L_v) / (liq->cp_liq(T_d_, p_sys) * m_d_);

    mdot_liq(i) = mdot_liq_;
  }
}