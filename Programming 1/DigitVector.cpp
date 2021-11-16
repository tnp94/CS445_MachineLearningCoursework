#include "DigitVector.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include "../eigen/Eigen/Core"

using namespace std;

int DigitVector::printDigit()
{
  cout << "Digit Label: " << label << "\n";
  for (int i = 0; i < 784; i++)
  {
    if (i % 28 == 0)
    {
      cout << "\n";
    }
    cout << setw(3) << data[i];
  }
  cout << "\n\n";
  return 0;
}

DigitVector::DigitVector()
{
  data = Eigen::RowVectorXf(785);
}