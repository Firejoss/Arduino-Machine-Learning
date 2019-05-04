#pragma once

using namespace std;

typedef float nn_double;

//#define DEBUG
#define DEBUG_MEMORY
#define LEARNING_RATE 1.0

#ifdef __arm__
// should use uinstd.h to define sbrk but Due causes a conflict
extern "C" char* sbrk(int incr);
#else  // __ARM__
extern char *__brkval;
#endif  // __arm__

#define sigmoid(x)			1.0 / (1.0 + (nn_double)exp(-(nn_double)(x)))
#define sigmoidPrime(x)		(nn_double)(sigmoid(x) * (1.0 - sigmoid(x)))

#define relu(x)				x > 0 ? (nn_double) x : 0
#define reluPrime(x)		x > 0 ? (nn_double) 1.0 : 0//(1.0 - (nn_double)exp(x))

typedef nn_double(*activFn)(nn_double, bool);


struct Util {

	static nn_double sigmoidFn(nn_double x, bool isDerivative = false) {
		return isDerivative ? sigmoidPrime(x) : sigmoid(x);
	}

	static nn_double reluFn(nn_double x, bool isDerivative = false) {
		return isDerivative ? reluPrime(x) : relu(x);
	}

	static nn_double sumabs(vector<nn_double> const& v1) {

		nn_double sum = 0;
		for (auto& val : v1) {
			sum += val > 0 ? val : -1 * val;
		}
		return sum;
	}

	// returns the transpose of a "matrix" (m, n) => (n, m)
	static int transpose(vector<vector<nn_double>> const& v1, vector<vector<nn_double>> &v1Transp) {

		if (v1.empty()) {
			Util::printMsg("Nothing to transpose.");
			return 1;
		}

		v1Transp.resize(v1[0].size());

		int columnSize = v1.size();
		for (auto& column : v1Transp) {
			column.resize(columnSize);
		}

		for (u_int32_t i = 0; i < v1.size(); i++)
		{
			for (u_int32_t j = 0; j < v1[i].size(); j++)
			{
				v1Transp[j][i] = v1[i][j];
			}
		}
		return 0;
	}

	// computes the dot product of two vectors
	static nn_double dot(vector<nn_double> v1, vector<nn_double> v2) {

		if (v1.size() != v2.size()) {
			printMsgInts("Dot product error. Unequal vectors sizes : ", { v1.size(), v2.size() });
			return 0;
		}

		nn_double result = 0;
		for (int i = 0, size = v1.size(); i < size; i++) {
			result += v1[i] * v2[i];
		}
		return result;
	}

	static void printMsg(String msg) {
		Serial.println(msg);
	}

	static void printMsgFloat(String msg, nn_double argument) {
		Serial.print(msg);
		Serial.println(argument, 5);
	}

	static void printMsgFloats(String msg, vector<nn_double> arguments) {
		Serial.print(msg);
		for (auto argument : arguments) {
			Serial.print(argument, 5);
			Serial.print(", ");
		}
		Serial.print("\n");
	}

	static void printMsgInt(String msg, int argument) {
		Serial.print(msg);
		Serial.println(argument);
	}

	static void printMsgInts(String msg, vector<int> arguments) {
		Serial.print(msg);
		for (auto argument : arguments) {
			Serial.print(argument);
			Serial.print(", ");
		}
		Serial.print("\n");
	}
};

struct TrainingSet {

	vector<nn_double> inputValues;
	vector<nn_double> idealOutputValues;

	TrainingSet()
	{}

	TrainingSet(int inputSize_, int idealOutputSize_)
	{
		inputValues.resize(inputSize_);
		idealOutputValues.resize(idealOutputSize_);
	}

	TrainingSet(nn_double inputs_[], int inputSize_, vector<nn_double> const& desiredOutput_) :
		idealOutputValues(desiredOutput_)
	{
		inputValues.insert(inputValues.begin(), inputs_, inputs_ + inputSize_);
	}

	TrainingSet(vector<nn_double> const& inputs_, vector<nn_double> const& desiredOutput_) :
		inputValues(inputs_),
		idealOutputValues(desiredOutput_)
	{}
};

struct Memory {

	static int getFreeMemory() {
		char top;
#ifdef __arm__
		return &top - reinterpret_cast<char*>(sbrk(0));
#elif defined(CORE_TEENSY) || (ARDUINO > 103 && ARDUINO != 151)
		return &top - __brkval;
#else  // __arm__
		return __brkval ? &top - __brkval : &top - __malloc_heap_start;
#endif  // __arm__
	}
};