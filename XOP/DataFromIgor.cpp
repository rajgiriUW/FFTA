#include <FPGAHeader.h>
#include <NiFpga.h>
#include <XOPStandardHeaders.h>		// Include ANSI headers, Mac headers, IgorXOP.h, XOP.h and XOPSupport.h
#include "trEFMAnalysisPackage.h"

static int gCallSpinProcess = 1;

double** ImportDataFromIgor(waveHndl data_wave,int* rows,int*columns)
{
	/*
	Gets data from an igor wave for use in our analysis.
	*/

	waveHndl						wavH;
	double							*dp0, *dcp, *dp;
	double**						 data;
	int								 numDimensions;
	CountInt						dimensionSizes[MAX_DIMENSIONS + 1];
	CountInt						numRows, numColumns, numLayers;
	CountInt						column;
	IndexInt						row;
	BCInt							numBytes;
	double*							dPtr;
	int								result, result2;
	char noticeStr[50];

	wavH = data_wave;

	MDGetWaveDimensions(wavH, &numDimensions, dimensionSizes);

	numRows = dimensionSizes[0];
	numColumns = dimensionSizes[1];
	numLayers = 0; // 2d waves

	// let the caller know what these dimensions are.
	(*rows) = numRows;
	(*columns) = numColumns;

	// Create the 2 dimensional data array.
	const int nrow = (int)numRows, ncol = (int)numColumns, nelem = nrow*ncol;
	data = new double*[nrow];
	data[0] = new double[nelem];
	for (int i = 1; i < nrow; i++)
	{
		data[i] = data[i - 1] + ncol;
	}

	numBytes = WavePoints(wavH) * sizeof(double);			// Bytes needed for copy
	dPtr = (double*)NewPtr(numBytes);
	MDGetDPDataFromNumericWave(wavH, dPtr); 	// Get a copy of the wave data.

	dp0 = dPtr;
	for (column = 0; column < numColumns; column++) {
		if (gCallSpinProcess && SpinProcess()) {		// Spins cursor and allows background processing.
			result = -1;								// User aborted.
			break;
		}
		dcp = dp0 + column*numRows;				// Pointer to start of data for this column.
		for (row = 0; row < numRows; row++) {
			dp = dcp + row;
			(double)data[row][column] = (*dp);
		}
	}
	DisposePtr((Ptr)dPtr);

	return data;



}