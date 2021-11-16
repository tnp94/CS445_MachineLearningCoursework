#ifndef DIGITVECTOR_H
#define DIGITVECTOR_H
#include "../eigen/Eigen/Core"

class DigitVector{
  public:
    int label;
    Eigen::VectorXf data;
    int printDigit();
    DigitVector();
};
#endif