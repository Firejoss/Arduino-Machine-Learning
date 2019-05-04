#pragma once

#include "NeuralNetworkUtils.h"

int buildTrainingSetString(String& trainSetStr_, TrainingSet& trainingSet_) {

	if (trainingSet_.inputValues.empty() || trainingSet_.idealOutputValues.empty()) {
		Util::printMsg("buildTrainingSetString() : training set incomplete. String building aborted.");
		return -1;
	}

	for (auto& inputVal : trainingSet_.inputValues) {
		trainSetStr_.append(String(inputVal, 12));
		trainSetStr_.append(' ');
	}
	trainSetStr_.append('#');
	for (auto& outputIdealVal : trainingSet_.idealOutputValues) {
		trainSetStr_.append(String(outputIdealVal, 12));
	}
	return 0;
}

int saveTrainingSetSDCard(TrainingSet& trainingSet_) {

	String trainingSetStr = "";

	// open the file. note that only one file can be open at a time,
	// so you have to close this one before opening another.
	File dataFile = SD.open(FILENAME_TRAIN_DATA, FILE_WRITE);

	if (dataFile)
	{
		if (-1 == buildTrainingSetString(trainingSetStr, trainingSet_)) {
			return -1;
		}

		dataFile.println(trainingSetStr);
		dataFile.close();
		return 0;
	}
#ifdef DEBUG_SDCARD
	String msg = "saveTrainingSetSDCard() : error opening file ";
	Util::printMsg(msg.append(FILENAME_TRAIN_DATA));
#endif
	return -1;
}

int readNextTrainingSetSDCard(TrainingSet& trainingSet_) {

	if (!SD.exists(FILENAME_TRAIN_DATA)) {
#ifdef DEBUG_SDCARD
		Util::printMsg("readNextTrainingSetSDCard() : file does not exist. Abort.");
#endif
		return -1;
	}

	// open the file. note that only one file can be open at a time,
	// so you have to close this one before opening another.
	File dataFile = SD.open(FILENAME_TRAIN_DATA, FILE_READ);

	return 0;
}

int deleteTrainingDataSDCardFile() {

	if (!SD.exists(FILENAME_TRAIN_DATA)) {
#ifdef DEBUG_SDCARD
		Util::printMsg("deleteTrainingDataSDCardFile() : file does not exist.");
#endif
		return 0;
	}
	return SD.remove(FILENAME_TRAIN_DATA);
}
