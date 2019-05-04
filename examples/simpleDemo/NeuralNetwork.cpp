
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(int inputVectorSize_, vector<int> intermediateLayersSizes_, int ouputVectorSize_)
{
	intermediateLayersSizes_.insert(intermediateLayersSizes_.begin(), inputVectorSize_);
	intermediateLayersSizes_.push_back(ouputVectorSize_);
	init(intermediateLayersSizes_);
}

NeuralNetwork::~NeuralNetwork()
{
}

int NeuralNetwork::init(vector<int> layersSizes_) {

	// --- ACTIVATION FUNCTIONS ---
	layersActivationFn = &Util::sigmoidFn;
	outputActivationFn = &Util::sigmoidFn;

	// --- INPUT, INTERMEDIATE AND OUTPUT LAYERS 
	weights.resize(layersSizes_.size() - 1);
	biases.resize(layersSizes_.size() - 1);
	zs.resize(layersSizes_.size() - 1);

	// errors vector is same size as output vector
	errors.resize(layersSizes_.back());

	deltas.resize(layersSizes_.size() - 1);
	
	neuronOutputs.resize(layersSizes_.size());
	neuronOutputs.front().resize(layersSizes_.front());

	for (int i = 1; i < layersSizes_.size(); i++) {
		neuronOutputs[i].resize(layersSizes_[i]);
		deltas[i - 1].resize(layersSizes_[i]);
		
		zs[i - 1].resize(layersSizes_[i]);
		biases[i - 1].resize(layersSizes_[i]);
		weights[i - 1].resize(layersSizes_[i]);

		for (int k = 0; k < layersSizes_[i]; k++) {
			weights[i - 1][k].resize(layersSizes_[i - 1]);
		}
	}

	randomizeWeights();
	randomizeBiases();

	Util::printMsg("Neural network initialized !");

	return 0;
}

int NeuralNetwork::randomizeWeights() {

	// weights ordered by layers of neurons
	for (auto& layer : weights) {

		// sublayers containing the weights of the connections
		// between one neuron and each neuron of the previous layer
		for (auto& subLayer : layer) {

			// every weight initialized randomly in [-0.99 ; 0.99]

			for (auto& weight : subLayer) {

				weight = (nn_double)(rand() % 200 - 100) / 100.0;

			}
		}
	}
	return 0;
}

int NeuralNetwork::randomizeBiases() {

	// biases ordered by layer of neurons
	for (auto& layer : biases) {

		// every weight initialized randomly in [-0.99 ; 0.99]
		for (auto& bias : layer) {

			bias = (nn_double)(rand() % 200 - 100) / 100.0;
		}
	}
	return 0;
}

nn_double NeuralNetwork::train(vector<TrainingSet> & trainingData_, nn_double idealError, u_int maxEpochs) {

	nn_double error;
	u_int numEpochs = 0;

#ifdef DEBUG
	Serial.println("\n--- Starting NN training ---");
#endif

	do {
		numEpochs++;
		Util::printMsgInt("\nEpoch ", numEpochs);

		error = 0;

		random_shuffle(begin(trainingData_), end(trainingData_));

		for (auto& trainingSet : trainingData_) {

#ifdef DEBUG
			Serial.println("Feed inputs...");
#endif
			feedInputs(trainingSet);
#ifdef DEBUG
			Serial.println("Propagate...");
#endif
			propagate();

#ifdef DEBUG
			Serial.println("Process error...");
#endif
			feedOutputIdealValues(trainingSet);
			error += Util::sumabs(errors);

			backpropagate();
		}
		error /= trainingData_.size();
//#ifdef DEBUG
		Util::printMsgFloat("Error => ", error);
//#endif

	} while (error > idealError && numEpochs < maxEpochs);

	return error;
}

int NeuralNetwork::feedInputs(TrainingSet &trainingSet) {

	if (trainingSet.inputValues.size() != neuronOutputs.front().size()) {

		Util::printMsgInts("trainingset and input vectors have different sizes : ", 
			{ trainingSet.inputValues.size(), neuronOutputs.front().size() });
		return -1;
	}

	int inputSize = neuronOutputs.front().size();
	for (int i = 0; i < inputSize; i++) {
		neuronOutputs.front()[i] = trainingSet.inputValues[i];
	}

#ifdef DEBUG
	Serial.print("\n--- input values : ---\n\n--> ");
	for (auto &inputVal : neuronOutputs.front()) {
		Serial.print(inputVal);
		Serial.print(" | ");
	}
	Serial.print("\n\n");
#endif

	return 0;
}

int NeuralNetwork::propagate() {

	int layersNum = neuronOutputs.size();

	// begin at 1 because index 0 is the input layer;
	for (int i = 1; i < layersNum; i++) {

		for (int j = 0; j < neuronOutputs[i].size(); j++) {

			nn_double sum = 0;
			for (int k = 0; k < neuronOutputs[i - 1].size(); k++) {
				sum += neuronOutputs[i - 1][k] * weights[i - 1][j][k];
			}

			zs[i - 1][j] = sum + biases[i - 1][j];

			if (i != layersNum - 1) {
				neuronOutputs[i][j] = layersActivationFn(zs[i - 1][j], false);
				continue;
			}
			neuronOutputs[i][j] = outputActivationFn(zs[i - 1][j], false);
		}
	}
#ifdef DEBUG
	printOutput();
#endif
	return 0;
}

vector<nn_double>& NeuralNetwork::feedOutputIdealValues(TrainingSet &trainingSet) {

	if (trainingSet.idealOutputValues.size() != neuronOutputs.back().size()) {
		Util::printMsg("Ideal output and output vectors have different sizes");
		return;
	}

	int outputNum = neuronOutputs.back().size();

#ifdef DEBUG
	Util::printMsgFloats("Feeding IDEAL values => ", trainingSet.idealOutputValues);
#endif

	for (int i = 0; i < outputNum; i++) {
		errors[i] = neuronOutputs.back()[i] - trainingSet.idealOutputValues[i];
	}
	return errors;
}

int NeuralNetwork::backpropagate() {

	//-- first initating backward pass --
	for (int i = 0; i < errors.size(); i++) {

		deltas.back()[i] = errors[i] * outputActivationFn(zs.back()[i], true)
									 * LEARNING_RATE;

		biases.back()[i] -= deltas.back()[i];

		for (int j = 0; j < weights.back()[i].size(); j++) {

			weights.back()[i][j] -= deltas.back()[i] * neuronOutputs[neuronOutputs.size() - 2][j];
		}
	}
	// ----------------------------------

	for (int k = deltas.size() - 2; k >= 0; k--) {

		vector<vector<nn_double>> transpWeights;
		Util::transpose(weights[k + 1], transpWeights);

		for (int l = 0; l < transpWeights.size(); l++) {

			deltas[k][l] = Util::dot(deltas[k + 1], transpWeights[l]) 
							* layersActivationFn(zs[k][l], true) 
							* LEARNING_RATE;
		}

		for (int m = 0; m < weights[k].size(); m++) {

			biases[k][m] -= deltas[k][m];

			for (int p = 0; p < weights[k][m].size(); p++) {

				weights[k][m][p] -= deltas[k][m] * neuronOutputs[k][p];
			}
		}

	}
#ifdef DEBUG
	Util::printMsg("backpropagation done.");
#endif

	return 0;
}

void NeuralNetwork::printOutput() {

	Serial.print("\n--- Neural Network Output : | ");

	for (auto &outputVal : neuronOutputs.back()) {
		Serial.print(outputVal);
		Serial.print(" | ");
	}
	Serial.print("\n");
}
