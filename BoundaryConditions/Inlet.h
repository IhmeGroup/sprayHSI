//
// Created by Danyal Mohaddes on 3/11/20.
//

#ifndef HIEMENZ_SOLVER_INLET_H
#define HIEMENZ_SOLVER_INLET_H


#include "BCGeneric.h"

class Inlet : public BCGeneric {
private:
    std::string Type = "Inlet";
public:
    void Update() override;
};


#endif //HIEMENZ_SOLVER_INLET_H
