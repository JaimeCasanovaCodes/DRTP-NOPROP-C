#ifndef DRTP_H
#define DRTP_H

#include "neural_network.h"

// DRTP-specific function declarations
void AllocateDRTPBuffers(NeuralNetwork* nn);
int GenerateRandomTargets(NeuralNetwork* nn, double* target);
int UpdateWeights(NeuralNetwork* nn, double* input, double* target);

// Hybrid approach constants
#define USE_NO_PROP_OUTPUT 1  // Set to 1 to use No Prop for output layer
#define LearningRate  0.1  
#define TargetScale   1.0     
#define MaxError      200.0    

#endif // DRTP_H 