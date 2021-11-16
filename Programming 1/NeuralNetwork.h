// Thomas Pollard
// CS445 - Programming 1
// NeuralNetwork.h
// This code uses the Eigen library which allows matrix manipulation and arithmetic
// Matrix * Matrix is matrix multiplication
// Vector * Vector is the dot product
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include "../eigen/Eigen/Core"

class NeuralNetwork{
  public:
    Eigen::VectorXf data;
    Eigen::Matrix<float, 785, Eigen::Dynamic, Eigen::RowMajor> hiddenWeights, lastHiddenChange; // Weights from input (row) to hidden node (col) as well as the change from last step
    Eigen::Matrix<float, Eigen::Dynamic, 10, Eigen::RowMajor> outputWeights, lastOutputChange; // Weights from the hidden nodes (row) to output node (col) as well as the change from last step
    int recall(Eigen::RowVectorXf input); // A function that returns the predicted label after forward propogating
    Eigen::RowVectorXf forwardPropogate(Eigen::RowVectorXf inputVector); // A function that forward propogates and returns the final activation vector
    
    int train(Eigen::RowVectorXf inputVector, int label, float eta);
  
    NeuralNetwork(int hiddenNodes, float inMomentum);
  private:
    void sigmoidActivation(Eigen::RowVectorXf &activationVector); // A sigmoid activation on the passed vector
    int hiddenNodeCount;
    float momentum;
    int hiddenWeightRows, hiddenWeightCols, outputWeightRows, outputWeightCols; // Number of corresponding rows/cols of matrices
};
#endif
