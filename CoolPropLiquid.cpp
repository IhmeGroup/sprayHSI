//
// Created by Danyal Mohaddes on 1/28/21.
//

#include <map>
#include <iostream>
#include <vector>
#include "boost/algorithm/string.hpp"
#include "CoolProp.h"
#include "CoolPropLiquid.h"

CoolPropLiquid::CoolPropLiquid(std::string X_liq_) : Liquid(X_liq_), coolprop_string(GetCoolPropString(X_liq_)) {}

std::string CoolPropLiquid::GetCoolPropName(const std::string cantera_name){
  std::map<std::string, std::string> cantera_to_CoolProp = {
          {"NC12H26","NC12H26"},
          {"N-C12H26","NC12H26"},
          {"KERO_LUCHE","NC12H26"}
  };
  // check if it's in the map, if not throw an error. CoolProp fails silently if it doesn't recognize the name
  if (cantera_to_CoolProp.count(cantera_name)){
    return cantera_to_CoolProp[cantera_name];
  } else {
    std::cerr << "No known conversion of Cantera species " << cantera_name << " to a CoolProp species." << std::endl;
    throw(0);
  }
}

std::string CoolPropLiquid::GetCoolPropString(std::string cantera_string) {
  std::string cool_prop_name = "";
  // Start with something like "A:0.7,B:0.3"
  std::vector<std::string> name_val_pairs;
  boost::split(name_val_pairs, cantera_string, [](char c) { return c == ','; });
  // Now have ["A:0.7","B:0.3"]
  for (auto &pair : name_val_pairs) {
    // Start with "A:0.7"
    std::vector<std::string> split_pair;
    boost::split(split_pair, pair, [](char c) { return c == ':'; });
    // Now have ["A","0.7"]
    split_pair[0] = GetCoolPropName(split_pair[0]);
    // Now have ["a","0.7"]
    pair = split_pair[0] + "[" + split_pair[1] + "]";
    // Now have "a[0.7]"
    cool_prop_name += pair + "&";
  }
  cool_prop_name = cool_prop_name.substr(0, cool_prop_name.length() - 1);
  return cool_prop_name;
}

double CoolPropLiquid::T_sat(double p) const {
  return CoolProp::PropsSI("T", "P", p, "Q", 0.0, coolprop_string);
}

double CoolPropLiquid::p_sat(double T) const {
  return CoolProp::PropsSI("P", "T", T, "Q", 0.0, coolprop_string);
}

double CoolPropLiquid::L_v(double p) const {
  return CoolProp::PropsSI("HMASS", "P", p, "Q", 1.0, coolprop_string) -
         CoolProp::PropsSI("HMASS", "P", p, "Q", 0.0, coolprop_string);
}

double CoolPropLiquid::rho_liq(double T, double p) const {
  // Ensure to pick a liquid density; trust pressure more than temperature
  if (T > T_sat(p))
    return CoolProp::PropsSI("DMASS", "P", p, "Q", 0.0, coolprop_string);
  else
    return CoolProp::PropsSI("DMASS", "T", T, "P", p, coolprop_string);
}

double CoolPropLiquid::rho_satvap(double p) const {
  return CoolProp::PropsSI("DMASS", "P", p, "Q", 1.0, coolprop_string);
}

double CoolPropLiquid::cp_liq(double T, double p) const {
  // Ensure to pick a liquid cp; trust pressure more than temperature
  if (T > T_sat(p))
    return CoolProp::PropsSI("CPMASS", "P", p, "Q", 0.0, coolprop_string);
  else
    return CoolProp::PropsSI("CPMASS", "T", T, "P", p, coolprop_string);
}

double CoolPropLiquid::cp_satvap(double p) const {
  return CoolProp::PropsSI("CPMASS", "P", p, "Q", 1.0, coolprop_string);
}
