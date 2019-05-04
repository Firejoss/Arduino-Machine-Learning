#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <algorithm>
#include "Arduino.h"
#include "NeuralNetworkUtils.h"

class NeuralNetwork
{

protected:

	activFn 							layersActivationFn;
	activFn 							outputActivationFn;

	vector<vector<nn_double>>			neuronOutputs;
	vector<vector<nn_double>>			zs;			// neuron's inputs * weights + bias

	vector<vector<vector<nn_double>>>	weights;
	vector<vector<nn_double>>			biases;

	vector<vector<nn_double>>			deltas;
	vector<nn_double>					errors;		// computed substracting ideal values from outputs

public:
	NeuralNetwork();
	NeuralNetwork(int inputVectorSize, vector<int> intermediateLayersSizes, int ouputVectorSize);
	~NeuralNetwork();

	int init(vector<int> layersSizes_);

	int randomizeWeights();

	int randomizeBiases();

	nn_double train(vector<TrainingSet> & trainingData_, nn_double idealError, u_int maxEpochs);

	int feedInputs(TrainingSet & trainingInputValues);

	int propagate();

	vector<nn_double>& feedOutputIdealValues(TrainingSet & trainingSet);

	int backpropagate();

	void printOutput();

};

#endif // NEURAL_NETWORK_H
