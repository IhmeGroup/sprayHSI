//
// Created by Danyal Mohaddes on 3/11/20.
//

#ifndef HIEMENZ_SOLVER_BCGENERIC_H
#define HIEMENZ_SOLVER_BCGENERIC_H

#include <string>

class BCGeneric {
private:
    std::string Type = "Generic";
public:
    virtual void Update() = 0;

    std::string GetType();
};


#endif //HIEMENZ_SOLVER_BCGENERIC_H
