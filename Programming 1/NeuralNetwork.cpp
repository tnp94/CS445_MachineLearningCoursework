// Thomas Pollard
// CS445 - Programming 1
// NeuralNetwork.cpp
// This code uses the Eigen library which allows matrix manipulation and arithmetic
// Matrix * Matrix is matrix multiplication
// Vector * Vector is the dot product
#include <iostream>
#include <cmath>
#include "../eigen/Eigen/Core"
#include "NeuralNetwork.h"

using namespace std;

int NeuralNetwork::recall(Eigen::RowVectorXf inputVector)
{
  int output = -1;
  Eigen::RowVectorXf activationVector = forwardPropogate(inputVector);
  activationVector.maxCoeff(&output);

  return output;
}

Eigen::RowVectorXf NeuralNetwork::forwardPropogate(Eigen::RowVectorXf inputVector)
{
  Eigen::RowVectorXf activationVector = inputVector * hiddenWeights;
  Eigen::RowVectorXf hiddenInput(hiddenNodeCount + 1);
  //cout << "Attempting to recall on inputVector:\n" << inputVector << "\n\n";
  sigmoidActivation(activationVector);
  for (int i = 0; i < hiddenInput.size(); i++) // Place the output of the hiddenWeights products in a vector for the output layer to take
  {
    if (i == 0)
    {
      hiddenInput(i) = 1;
    }
    else
    {
      hiddenInput(i) = activationVector(i-1);
    }
  }
  //cout << "Output of the input -> hidden layer:\n" << hiddenInput << "\n\n";
  
  activationVector = hiddenInput * outputWeights;
  sigmoidActivation(activationVector);
  
  return activationVector;
}

void NeuralNetwork::sigmoidActivation(Eigen::RowVectorXf &activationVector)
{
  //cout << "sigmoidActivation called on:\n";
  //cout << activationVector;
  for (int i = 0; i < activationVector.size(); i++)
  {
    activationVector(i) = 1/(1+exp((-1*activationVector(i))));
  }
  //cout << "\nThe results:\n";
  //cout << activationVector << "\n";
  return;
}

NeuralNetwork::NeuralNetwork(int hiddenNodes, float inMomentum)
{
  
  hiddenWeights = Eigen::MatrixXf::Random(785,hiddenNodes); // Matrix of weights from the input (row) to the number of hidden nodes plus the bias
  lastHiddenChange = Eigen::MatrixXf::Zero(785,hiddenNodes);
  for (int i = 0; i < hiddenWeights.rows(); i++) // Initialize the hiddenWeights
  {
    for (int j = 0; j < hiddenWeights.cols(); j++)
    {
      hiddenWeights(i,j) = (((hiddenWeights(i,j) - -1) * (0.05 - -0.05)) / (1 - -1)) + -0.05; // Convert each weight to a min of -0.05 and a max of 0.05
    }
  }
  
  outputWeights = Eigen::MatrixXf::Random(hiddenNodes+1, 10); // Matrix of weights from the number of hidden nodes plus the bias to the output
  lastOutputChange =  Eigen::MatrixXf::Zero(hiddenNodes + 1, 10);
  for (int i = 0; i < outputWeights.rows(); i++) // Initialize the outputWeights
  {
    for (int j = 0; j < outputWeights.cols(); j++)
    {
      outputWeights(i,j) = (((outputWeights(i,j) - -1) * (0.05 - -0.05)) / (1 - -1)) + -0.05; // Convert each weight to a min of -0.05 and a max of 0.05
    }
  }
  
  hiddenWeightRows = hiddenWeights.rows();
  hiddenWeightCols = hiddenWeights.cols();
  outputWeightRows = outputWeights.rows();
  outputWeightCols = outputWeights.cols();
  hiddenNodeCount = hiddenNodes;
  momentum = inMomentum;
}

int NeuralNetwork::train(Eigen::RowVectorXf inputVector, int label, float eta)
{
  int success = -1;
  if (recall(inputVector) == label)
  {
    success = 0;
  }
  else
  {
    Eigen::VectorXf oneHotVector(10);
    for (int i = 0; i < oneHotVector.size(); i++)
    {
      oneHotVector(i) = 0.1;
    }
    oneHotVector(label) = 0.9;
    
    Eigen::VectorXf outputDeltas(10), hiddenDeltas(hiddenNodeCount+1);
    float activation;
    Eigen::RowVectorXf outputActivationVector = forwardPropogate(inputVector);
    
    for (int i = 0; i < outputDeltas.size(); i++) // Calculate the error terms for the output layer
    {
      activation = outputActivationVector(i);
      outputDeltas(i) = activation * (1 - activation) * (oneHotVector[i] - activation);
    }
    
    // Find the activation results of the hidden layer
    Eigen::RowVectorXf activationVector = inputVector * hiddenWeights;
    sigmoidActivation(activationVector);
    Eigen::RowVectorXf hiddenActivationVector(hiddenNodeCount + 1);
    for (int i = 0; i < hiddenActivationVector.size(); i++) // Now place the bias
    {
      if (i == 0)
      {
        hiddenActivationVector(i) = 1;
      }
      else
      {
        hiddenActivationVector(i) = activationVector(i-1);
      }
    }
    
    for (int i = 0; i < hiddenDeltas.size(); i++) // Calculate the error terms for the hidden layer
    {
      activation = hiddenActivationVector(i);
      float sum = 0;
      /*for (int j = 0; j < outputWeights.cols(); j++)
      {
        sum += outputWeights(i,j) * outputDeltas(j);
      }*/
      sum = outputWeights.row(i) * outputDeltas;
      hiddenDeltas(i) = activation * (1 - activation) * sum;
    }
    
    
    // Now I need to update weights for both layers
    float momentumChange = 0;
    float weightChange;
    for (int i = 0; i < outputWeightRows; i++) // Output layer
    {
      for (int j = 0; j < outputWeightCols; j++)
      {
        weightChange = eta * outputDeltas(j) * hiddenActivationVector(i);
        momentumChange = lastOutputChange(i,j) * momentum;
        weightChange += momentumChange;
        lastOutputChange(i,j) = weightChange;
        outputWeights(i,j) += weightChange;
      }
    }
    for (int i = 0; i < hiddenWeightRows; i++)
    {
      for (int j = 0; j < hiddenWeightCols; j++)
      {
        weightChange = eta * hiddenDeltas(j) * inputVector(i);
        momentumChange = lastHiddenChange(i,j) * momentum;
        weightChange += momentumChange;
        lastHiddenChange(i,j) = weightChange;
        hiddenWeights(i,j) += weightChange;
      }
    }
    success = 1; // This is to debug when I train something or not
  }
  
  return success;
}