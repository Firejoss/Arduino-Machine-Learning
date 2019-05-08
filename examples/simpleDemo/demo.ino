
#include "NeuralNetwork.h"

// prevent compile time error when using std::vector C++ library
namespace std {
	void __throw_bad_alloc()
	{
		Serial.println("Unable to allocate memory");
	}

	void __throw_length_error(char const*e)
	{
		Serial.print("Length Error :");
		Serial.println(e);
	}
}

#define DEBUG_SERIAL

// shape of the neural network (number of inputs, outputs and intermediate layers)
#define NN_INPUT_SIZE 5
#define NN_HIDDEN_LAYERS_SIZES { 6, 3 }
#define NN_OUTPUT_SIZE 1

NeuralNetwork * simpleNN;


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

	TrainingSet ts0(v0, { 0.5 });
	TrainingSet ts1(v1, { 0.0 });
	TrainingSet ts2(v2, { 1.0 });
	TrainingSet ts3(v3, { 0.7 });
	TrainingSet ts4(v4, { 1.5 });

	vector<TrainingSet> tsets{ ts0, ts1, ts2, ts3, ts4 };

	for (size_t i = 0; i < 600; i++)
	{
		simpleNN->feedInputs(tsets[i % 4]);
		simpleNN->propagate();
		simpleNN->feedOutputIdealValues(tsets[i % 4]);
		simpleNN->backpropagate();
	}
	Util::printMsg("\n************************");
	Util::printMsg("*** Training is over ***");
	Util::printMsg("************************\n");

	for (auto& tset : tsets) {
		simpleNN->feedInputs(tset);
		simpleNN->propagate();
		simpleNN->printOutput();
	}

	while (1);
}