
#include "NeuralNetwork.h"

#define DEBUG_SERIAL

// shape of the neural network (number of inputs, outputs and intermediate layers)
#define NN_INPUT_SIZE 5
#define NN_HIDDEN_LAYERS_SIZES { 6, 3 }
#define NN_OUTPUT_SIZE 1


//---------------------- NN SCHEMA ----------------------
//           | - O - |       
//  >> I0 -- |       |       
//           | - O - |       
//  >> I1 -- |       | - O - |
//           | - O - |       | \-
//  >> I2 -- |       | - O - | ----->> O (final Output)
//           | - O - |       | /-
//  >> I3 -- |       | - O - |
//           | - O - |       
//  >> I4 -- |       |       
//           | - O - |       
//-------------------------------------------------------


// ---------------------------------------------------
// ---------------------- SETUP ----------------------
// ---------------------------------------------------

void setup() {

#ifdef DEBUG_SERIAL
	Serial.begin(115200);
#endif

	// creates the neural network
	simpleNN = new NeuralNetwork(NN_INPUT_SIZE, NN_HIDDEN_LAYERS_SIZES, NN_OUTPUT_SIZE);

}


// --------------------------------------------------
// ---------------------- LOOP ----------------------
// --------------------------------------------------

void loop() {

	vector<nn_double> v0;
	vector<nn_double> v1;
	vector<nn_double> v2;
	vector<nn_double> v3;
	vector<nn_double> v4;
	for (size_t i = 0; i < NN_INPUT_SIZE; i++)
	{
		v0.push_back(((rand() % 200) - 100) / 100.0);
		v1.push_back(((rand() % 200) - 100) / 100.0);
		v2.push_back(((rand() % 200) - 100) / 100.0);
		v3.push_back(((rand() % 200) - 100) / 100.0);
		v4.push_back(((rand() % 200) - 100) / 100.0);
	}

	TrainingSet ts0(v0, { 0.0, 0.0 });
	TrainingSet ts1(v1, { 0.0, 1.0 });
	TrainingSet ts2(v2, { 1.0, 1.0 });
	TrainingSet ts3(v3, { });
	TrainingSet ts4(v4, { });

	vector<TrainingSet> tsets{ ts0, ts1, ts2, ts3, ts4 };

	for (size_t i = 0; i < 600; i++)
	{
		bongoNeuralNetwork->feedInputs(tsets[i % 3]);
		bongoNeuralNetwork->propagate();
		bongoNeuralNetwork->feedOutputIdealValues(tsets[i % 3]);
		bongoNeuralNetwork->backpropagate();
	}
	Util::printMsg("\n************************");
	Util::printMsg("*** Training is over ***");
	Util::printMsg("************************\n");

	for (size_t i = 0; i < tsets.size(); i++)
	{
		bongoNeuralNetwork->feedInputs(tsets[i]);
		bongoNeuralNetwork->propagate();
		bongoNeuralNetwork->printOutput();
	}


	while (1);
}