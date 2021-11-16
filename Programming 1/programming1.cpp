// Thomas Pollard
// CS445 - Programming 1
// programming1.cpp
// This code uses the Eigen library which allows matrix manipulation and arithmetic
// Matrix * Matrix is matrix multiplication
// Vector * Vector is the dot product
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <limits.h>
#include <fstream>
#include <vector>
#include <iomanip>
#include "DigitVector.h"
#include "NeuralNetwork.h"
#include "../eigen/Eigen/Core"
#define EPOCHS 50
#define ETA 0.1
#define MOMENTUM 0.9
#define HIDDEN_UNITS 100
#define TRAIN_SUBSET_SIZE 2000 // I was originally using the full input set (inputTrainMatrixRows) but it was taking too long to run. Linux would terminate my program around epoch 15 and I would get no results.

using namespace std;

vector<DigitVector> inputMatrix;
vector<DigitVector> testMatrix;

int readTrainData()
{
  DigitVector vec;
  Eigen::VectorXf readVector(785);
  char *digit = new char[4];
  int currentVector = 0;
  int num;
  fstream inFile;
  inFile.open("mnist_train.csv");
  while (inFile.get(digit,INT_MAX,','))//!inFile.fail())
  {
    //cout << "Attempting to read vector " << currentVector << "\n";
    //inFile >> num;
    vec.label = atoi(digit);
    for (int i = 0; i < 785; i++)
    {
      if (i == 0)
      {
        readVector(i) = 1;
      }
      else
      {
        inFile.ignore(INT_MAX,',');
        inFile >> num;
        readVector(i) = num;
        //num = i;
      }
    }
    if (inFile.fail())
    {
      cout << "Failbit set at vector " << currentVector << ", breaking out of loop\n\n";
      return currentVector;
    }
    inFile.ignore(INT_MAX,'\n');
    vec.data = readVector;
    inputMatrix.push_back(vec);
    currentVector++;
  }
  delete [] digit;
  return 0;
}

int readTestData()
{
  DigitVector vec;
  char *digit = new char[4];
  int currentVector = 0;
  int num;
  fstream inFile;
  inFile.open("mnist_test.csv");
  while (inFile.get(digit,INT_MAX,','))//!inFile.fail())
  {
    //cout << "Attempting to read vector " << currentVector << "\n";
    //inFile >> num;
    vec.label = atoi(digit);
    for (int i = 0; i < 785; i++)
    {
      if (i == 0)
      {
        vec.data(i) = 1;
      }
      else
      {
        inFile.ignore(INT_MAX,',');
        inFile >> num;
        vec.data(i) = num;
        //num = i;
      }
    }
    if (inFile.fail())
    {
      cout << "Failbit set at vector " << currentVector << ", breaking out of loop\n\n";
      return currentVector;
    }
    inFile.ignore(INT_MAX,'\n');
    testMatrix.push_back(vec);
    currentVector++;
  }
  delete [] digit;
  return 0;
}


int activationFunction(Eigen::VectorXf &activationVector)
{
  for (int i = 0; i < activationVector.rows(); i++)
  {
    if (activationVector(i) > 0)
    {
      activationVector(i) = 1;
    }
    else
    {
      activationVector(i) = 0;
    }
  }
  return 0;
}


void printMatrixInterval(int low, int high, int interval)
{
  cout << "Printing digit " << low << " to " << high << " printing every " << interval << " digit.\n\n";
  for (int i = low; i <= high; i+=interval)
  {
    cout << "Printing digit " << i << ":\n";
    inputMatrix[i].printDigit();
  }
}

void matrixRandomizer(Eigen::Matrix<float, Eigen::Dynamic, 785> &inputTrainMatrix, vector<int> &trainLabels)
{
  // For each input vector, swap it with a random input further into the matrix as well as it's corresponding label
  for (int i = 0; i < inputTrainMatrix.rows(); i++)
  {
    int newIndex;
    newIndex = rand() % (inputTrainMatrix.rows()-i) + i; // Pick a random number from current to any further index
    
    Eigen::RowVectorXf tempRow = inputTrainMatrix.row(newIndex);
    inputTrainMatrix.row(newIndex) = inputTrainMatrix.row(i);
    inputTrainMatrix.row(i) = tempRow;
    
    int itemp = trainLabels[newIndex];
    trainLabels[newIndex] = trainLabels[i];
    trainLabels[i] = itemp;
  }
  
}

int main(int argc, char *argv[])
{
  int correctTrain[EPOCHS+1], incorrectTrain[EPOCHS + 1], correctTest[EPOCHS+1], incorrectTest[EPOCHS + 1];
  Eigen::Matrix<float, Eigen::Dynamic, 785> inputTrainMatrix, inputTestMatrix;
  Eigen::Matrix<float, 785, 10> weightMatrix = Eigen::MatrixXf::Random(785,10);
  vector<int> trainLabels, testLabels;
  
  // ---------- This section just reads the data and places it into an inputTrainMatrix and inputTestMatrix -----
  // ---------- It also places the labels into vector trainLabels and testLabels
  int failedAt = readTrainData();
  if (failedAt)
  {
    cout << "Data failed to read from mnist_train.csv\n";
    cout << "Printing digit " << failedAt-1 << "\n";
    inputMatrix[failedAt-1].printDigit();
    return -1;
  }
  failedAt = readTestData();
  if (failedAt)
  {
    cout << "Data failed to read from mnist_test.csv\n";
    cout << "Printing digit " << failedAt-1 << "\n";
    testMatrix[failedAt-1].printDigit();
    return -1;
  }
  inputTrainMatrix.resize(inputMatrix.size(), 785);
  inputTestMatrix.resize(testMatrix.size(), 785);
  
  for (int i = 0; i < weightMatrix.rows(); i++) // Initialize the weightMatrix
  {
    for (int j = 0; j < weightMatrix.cols(); j++)
    {
      weightMatrix(i,j) = (((weightMatrix(i,j) - -1) * (0.05 - -0.05)) / (1 - -1)) + -0.05; // Convert each weight to a min of -0.05 and a max of 0.05
    }
  }
  
  for (uint i = 0; i < inputMatrix.size(); i++)
  {
    trainLabels.push_back(inputMatrix[i].label);
    inputTrainMatrix.row(i) = inputMatrix[i].data/255.0;
    inputTrainMatrix(i,0) = 1;
  }

  for (uint i = 0; i < testMatrix.size(); i++)
  {
    testLabels.push_back(testMatrix[i].label);
    inputTestMatrix.row(i) = testMatrix[i].data/255.0;
    inputTestMatrix(i,0) = 1; // THIS IS THE BIAS
  }
  int inputTrainMatrixRows = inputTrainMatrix.rows();
  int inputTestMatrixRows = inputTestMatrix.rows();
  // I was originally using the full input set (inputTrainMatrixRows) but it was taking too long to run. Linux would terminate my program around epoch 15 and I would get no results.
  // -------------- This is the end of the data reading portiong
  
  
  
  cout << "Eta: " << ETA << "\nMomentum: " << MOMENTUM << "\nTraining Subset Size: " << TRAIN_SUBSET_SIZE << "\nHidden Nodes: " << HIDDEN_UNITS << "\n";
  
  Eigen::RowVectorXf inputVector = inputTrainMatrix.row(0);
  cout << "Now initializing the mlp\n";
  NeuralNetwork mlp(HIDDEN_UNITS, MOMENTUM);
  int confusionMatrix[10][10] = {0};
  for (int epoch = 0; epoch <= EPOCHS; epoch++) // For each epoch
  {
    int correctCount = 0, incorrectCount = 0, guess = 0;
    //cout << "Starting Epoch " << epoch << "\n";
    if (epoch > 0) // If this is not the 0 epoch we need to train the weights first
    {
      cout << "Training...\n";
      for (int i = 0; i < TRAIN_SUBSET_SIZE; i++)
      {
        mlp.train(inputTrainMatrix.row(i), trainLabels[i], ETA);
      }
    }
    matrixRandomizer(inputTrainMatrix, trainLabels);
    
    cout << "Testing...\n";
    for (int i = 0; i < 10000; i++) // Now test it on the training inputs
    {
      guess = mlp.recall(inputTrainMatrix.row(i));
      
      if (guess == trainLabels[i])
      {
        correctCount++;
      }
      else
      {
        incorrectCount++;
      }
    }
    correctTrain[epoch] = correctCount;
    incorrectTrain[epoch] = incorrectCount;
    
    correctCount = 0;
    incorrectCount = 0;
    
    for (int i = 0; i < inputTestMatrixRows; i++) // Now test it on the test inputs
    {
      guess = mlp.recall(inputTestMatrix.row(i));
      
      if (guess == testLabels[i])
      {
        correctCount++;
      }
      else
      {
        incorrectCount++;
      }
      if (epoch == EPOCHS)
      {
        confusionMatrix[testLabels[i]][guess] += 1;
      }
    }
    
    correctTest[epoch] = correctCount;
    incorrectTest[epoch] = incorrectCount;
    //cout << "Results for the test set for epoch " << epoch << ":\nCorrect: " << correctCount << "\nIncorrect: " << incorrectCount << "\n\n"; 
  }

  cout << "\n----------------Results of the training---------------\n";
  cout << "Eta: " << ETA << "\n";
  cout << left << setw(11) << "EPOCH:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << i << " | ";
  }
  cout << left << setw(12) << "\nCorrect:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << correctTrain[i] << " | ";
  }
  cout << left << setw(12) << "\nIncorrect:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << incorrectTrain[i] << " | ";
  }
  cout << left << setw(12) << "\nAccuracy:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << setprecision(4) << 100*(correctTrain[i]/60000.0) << "%| ";
  }
    cout << "\n----------------Results of the testing---------------\n";
  cout << "Eta: " << ETA << "\n";
  cout << left << setw(11) << "EPOCH:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << i << " | ";
  }
  cout << left << setw(12) << "\nCorrect:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << correctTest[i] << " | ";
  }
  cout << left << setw(12) << "\nIncorrect:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << incorrectTest[i] << " | ";
  }
  cout << left << setw(12) << "\nAccuracy:";
  for (int i = 0; i <= EPOCHS; i++)
  {
    cout << setw(6) << right << setprecision(4) << 100*(correctTest[i]/10000.0) << "%| ";
  }
  
  cout << "\n\nConfusion Matrix:\n";
  for (int i = 0; i <= 10; i++)
  {
    if (i > 0)
      cout << right << setw(6) << i-1 << "| ";
    else
      cout << setw(6) << "";
  }
  cout << "\n";
  for (int i = 0; i < 10; i++)
  {
    cout << right << setw(6) << i;
    for (int j = 0; j < 10; j++)
    {
      cout << right << setw(6) << confusionMatrix[i][j] << "| ";
    }
    cout << "\n";
  }  
  
  
  cout << "\n\n";
  return 0;
}
