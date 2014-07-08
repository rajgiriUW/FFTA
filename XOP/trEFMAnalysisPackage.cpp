#include <FPGAHeader.h>
#include <NiFpga.h>
#include <XOPStandardHeaders.h>		// Include ANSI headers, Mac headers, IgorXOP.h, XOP.h and XOPSupport.h
#include "trEFMAnalysisPackage.h"
#include <NiFpga_125MSAcquire.h>
#include <chrono>
#include "Python.h"
// Global Variables

static int gCallSpinProcess = 1;		// Set to 1 to all user abort (cmd dot) and background processing.

//////// HELPER FUNCTIONS TO BE USED BY THE XOP FUNCTIONS \\\\\\\\\\
////////////////////////////////////////////////////////////////////
/*
Transfer the data from a 2d array represented by a pointer-to-pointer type to a wave already existing
in IGOR. The size of data MUST match the size of wave or you will be in trouble, mister.
*/
void TransferToIgor(waveHndl wave, double** data)


{
	int								result,result2;
	int								 waveType;
	double							*dp0, *dcp, *dp;
	CountInt						numRows, numColumns, numLayers;
	int								numDimensions;
	CountInt						dimensionSizes[MAX_DIMENSIONS + 1];
	BCInt							numBytes;
	double*							dPtr;


	MDGetWaveDimensions(wave, &numDimensions, dimensionSizes);


	numRows = dimensionSizes[0];
	numColumns = dimensionSizes[1];
	numLayers = 0;

	numBytes = WavePoints(wave) * sizeof(double);			// Bytes needed for copy

	dPtr = (double*)NewPtr(numBytes);

	MDGetDPDataFromNumericWave(wave, dPtr);	// Get a copy of the wave data.

	result = 0;
	dp0 = dPtr;

	// Pointer to start of data for this layer.
	for (int column = 0; column < 1; column++) {
		if (gCallSpinProcess && SpinProcess()) {		// Spins cursor and allows background processing.
			result = -1;								// User aborted.
			break;
		}
		dcp = dp0 + column*numRows;				// Pointer to start of data for this column.
		for (int row = 0; row < numRows; row++) {
			dp = dcp + row;
			*dp = (double)data[row][column];
		}
	}

	MDStoreDPDataInNumericWave(wave, dPtr);	// Store copy in the wave.

	DisposePtr((Ptr)dPtr);

	WaveHandleModified(wave);
}
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
#pragma pack(2)		// All structures passed to Igor are two-byte aligned
struct AnalysisConfigParams {
	double result;
};
typedef struct AnalysisConfigParams AnalysisConfigParams;
#pragma pack()		// Reset structure alignment to default.

extern "C" int
AnalysisConfig(AnalysisConfigParams* p)
{
	char noticeStr[50];
	waveHndl			ConfigHandlePtr; // Place to put waveHndl for new wave
	char*				AcquisitionWaveName = "PIXELCONFIG\0"; // C string containing name of Igor wave
	int					SetCurrentDataFolder(DataFolderHandle dataFolderH);
	long				refNum = 0;
	int					result = 0;
	DataFolderHandle	rootDataFolderHPtr;
	DataFolderHandle	packageDataFolderH;
	DataFolderHandle	currentFolderH;
	char*				newDataFolderName;
	DataFolderHandle	newDataFolderHPtr;

	waveHndl			AcquisitionHandlePtr; // Place to put waveHndl for new wave
	int					FolderID;
	int					WaveType = NT_I32;
	
	GetCurrentDataFolder(&currentFolderH);

	newDataFolderName = "Analysis Config\0";

	if (result = GetRootDataFolder(0, &rootDataFolderHPtr))
	{
		return result;
	}

	if (result = GetNamedDataFolder(rootDataFolderHPtr, ":packages:", &packageDataFolderH))
	{
		result = NewDataFolder(rootDataFolderHPtr, newDataFolderName, &newDataFolderHPtr);
	}
	else {
		result = NewDataFolder(packageDataFolderH, newDataFolderName, &newDataFolderHPtr);
	}

	GetDataFolderIDNumber(newDataFolderHPtr, &FolderID);

	SetCurrentDataFolder(newDataFolderHPtr);

	if (result = MakeWave(&AcquisitionHandlePtr, AcquisitionWaveName, 7, NT_FP64, 0))
	{
		return result;
	}


	BCInt							numBytes;
	double*							dPtr;
	numBytes = WavePoints(AcquisitionHandlePtr) * sizeof(double);			// Bytes needed for copy

	dPtr = (double*)NewPtr(numBytes);
	dPtr[0] = 5e-4;
	dPtr[1] = 1e-3;
	dPtr[2] = 10e6;
	dPtr[3] = 300e3;
	dPtr[4] = 1;
	dPtr[5] = 1;
	dPtr[6] = 10e3;
	MDStoreDPDataInNumericWave(AcquisitionHandlePtr, dPtr);	// Store copy in the wave.

	DisposePtr((Ptr)dPtr);

	MDSetDimensionLabel(AcquisitionHandlePtr, 0, 0, "trigger\0");
	MDSetDimensionLabel(AcquisitionHandlePtr, 0, 1, "total_time\0");
	MDSetDimensionLabel(AcquisitionHandlePtr, 0, 2, "sampling_rate\0");
	MDSetDimensionLabel(AcquisitionHandlePtr, 0, 3, "drive_freq\0");
	MDSetDimensionLabel(AcquisitionHandlePtr, 0, 4, "window\0");
	MDSetDimensionLabel(AcquisitionHandlePtr, 0, 5, "bandpass_filter\0");
	MDSetDimensionLabel(AcquisitionHandlePtr, 0, 6, "filter_bandwidth\0");

	SetCurrentDataFolder(currentFolderH);

	return result;
}
#pragma pack(2)		// All structures passed to Igor are two-byte aligned
struct AnalyzeLineParams {
	waveHndl data_wave; // data collected goes here
	double recordLength;
	double preTriggerSamples;
	double numRecords;
	double numOfPixels;
	waveHndl tfp_wave;
	waveHndl shift_wave;
	waveHndl config_wave;
	double result;

};
typedef struct AnalyzeLineParams AnalyzeLineParams;
#pragma pack()		// Reset structure alignment to default.

extern "C" int
AnalyzeLine(AnalyzeLineParams* p)
{
	int averages = 15;
	int result = 0;
	int numOfPixels = (int)p->numOfPixels;
	char noticeStr[50];
	waveHndl wavH = p->data_wave;
	waveHndl tfp_wave = p->tfp_wave;
	waveHndl shift_wave = p->shift_wave;
	waveHndl config_wave = p->config_wave;
	double							trigger;
	double							total_time;
	double							sampling_rate;
	double							drive_freq;
	double							window;
	double							bandpass_filter;
	double							filter_bandwidth;
	uint32_t						numRecords = (uint32_t)p->numRecords;
	uint32_t						recordLength = (uint32_t)p->recordLength;
	int32_t							preTriggerSamples = (int32_t)p->preTriggerSamples;
	double*							configPtr;
	

	// Get Analysis settings from configuration wave.
	BCInt numBytes;
	configPtr = (double*)NewPtr(numBytes);
	MDGetDPDataFromNumericWave(config_wave, configPtr);	
	trigger = configPtr[0];
	total_time = configPtr[1];
	sampling_rate = configPtr[2];
	drive_freq = configPtr[3];
	window = configPtr[4];
	bandpass_filter = configPtr[5];
	filter_bandwidth = configPtr[6];


	/*
	#
	#        HERE WE DO EVERYTHING WE NEED TO DO IN ORDER TO FILL A WAVE WITH TFP DATA.
	#
	*/
	double tfp, shift;
	tfp = 0.0;
	shift = 0.0;

	// Acquire deflection data using NI 5762 FPGA digitizer
	double** data = FPGAAcquireData(numRecords, recordLength, preTriggerSamples);


	// set up tfp data
	double** tfp_data;
	double** shift_data;
	const int nrow = (int)numOfPixels, ncol = 1, nelem = nrow*ncol;
	tfp_data = new double*[nrow];
	shift_data = new double*[nrow];
	tfp_data[0] = new double[nelem];
	shift_data[0] = new double[nelem];
	for (int i = 1; i < nrow; i++)
	{
		tfp_data[i] = tfp_data[i - 1] + ncol;
		shift_data[i] = shift_data[i - 1] + ncol;
	}

	// Temp storage for data in one pixel.
	double** temp_data;
	const int temp_nrow = (int)recordLength, temp_ncol = averages, temp_nelem = temp_nrow*temp_ncol;
	temp_data = new double*[temp_nrow];
	temp_data[0] = new double[temp_nelem];
	for (int i = 1; i < temp_nrow; i++)
	{
		temp_data[i] = temp_data[i - 1] + temp_ncol;
	}

	// Analyze each pixel in the line and get the tfp data
	for (int pixel = 0; pixel < numOfPixels; pixel++)
	{
		int temp_column = 0;
		for (int column = pixel*averages; column < (pixel*averages + averages) ; column++)
		{
			int temp_row = 0;	
			for (int row = 0; row < recordLength; row++)
			{
				temp_data[temp_row][temp_column] = data[row][column];
				temp_row++;
			}
			temp_column++;
		}

		AnalyzePixel(&tfp, &shift, trigger, total_time, sampling_rate, drive_freq, window, bandpass_filter, filter_bandwidth, temp_data, recordLength, averages);
		tfp_data[pixel][0] = tfp;
		shift_data[pixel][0] = shift;
	}
	DisposePtr((Ptr)configPtr);
	/* BEGIN TRANSFER OF DATA TO IGOR HERE.*/

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	TransferToIgor(tfp_wave, tfp_data);
	TransferToIgor(shift_wave, shift_data);
	TransferToIgor(wavH, data);

	delete[] data[0];
	delete[] data;
	delete[] tfp_data[0];
	delete[] tfp_data;
	delete[] shift_data[0];
	delete[] shift_data;
	delete[] temp_data;

	return result;
}


#pragma pack(2)		// All structures passed to Igor are two-byte aligned
struct AnalyzeLineOfflineParams {
	waveHndl w;       // handle to igor wave containing data.
	waveHndl tfpWave;
	waveHndl shiftWave;
	double averages;
	waveHndl config_wave;
	double result;

};
typedef struct AnalyzeLineParams AnalyzeLineParams;
#pragma pack()		// Reset structure alignment to default.

extern "C" int
AnalyzeLineOffline(AnalyzeLineOfflineParams* p)
{
	int averages = (int)p->averages;
	char noticeStr[50];
	waveHndl wavH = p->w;
	waveHndl tfpWave = p->tfpWave;
	waveHndl shiftWave = p->shiftWave;
	waveHndl config_wave = p->config_wave;
	double							trigger;
	double							total_time;
	double							sampling_rate;
	double							drive_freq;
	double							window;
	double							bandpass_filter;
	double							filter_bandwidth;
	double*							configPtr;
	int numRows, numColumns;
	int result;

	// Get Analysis settings from configuration wave.
	BCInt numBytes;
	configPtr = (double*)NewPtr(numBytes);
	MDGetDPDataFromNumericWave(config_wave, configPtr);
	trigger = configPtr[0];
	total_time = configPtr[1];
	sampling_rate = configPtr[2];
	drive_freq = configPtr[3];
	window = configPtr[4];
	bandpass_filter = configPtr[5];
	filter_bandwidth = configPtr[6];


	double tfp, shift;
	tfp = 0.0;
	shift = 0.0;

	// Acquire data from Igor
	double** data = ImportDataFromIgor(wavH,&numRows,&numColumns);

	int numOfPixels = (int)(numColumns / averages);

	// set up tfp and shift data pointers
	double** tfp_data;
	double** shift_data;
	const int nrow = (int)(numColumns / averages), ncol = 1, nelem = nrow*ncol;
	tfp_data = new double*[nrow];
	tfp_data[0] = new double[nelem];
	shift_data = new double*[nrow];
	shift_data[0] = new double[nelem];
	for (int i = 1; i < nrow; i++)
	{
		tfp_data[i] = tfp_data[i - 1] + ncol;
		shift_data[i] = shift_data[i - 1] + ncol;
	}

	// For each pixel, create an array npoints X nAverages to store the data for a single pixel.
	// This data is then analyzed by calling the Python package, and a TFP value is stored and transfered
	// to igor.
	double** temp_data;
	const int temp_nrow = (int)numRows, temp_ncol = averages, temp_nelem = temp_nrow*temp_ncol;
	temp_data = new double*[temp_nrow];
	temp_data[0] = new double[temp_nelem];
	for (int i = 1; i < temp_nrow; i++)
	{
		temp_data[i] = temp_data[i - 1] + temp_ncol;
	}

	for (int pixel = 0; pixel < numOfPixels; pixel++)
	{
		int temp_column = 0;
		for (int column = pixel*averages; column < (pixel*averages + averages); column++)
		{
			int temp_row = 0;
			for (int row = 0; row < numRows; row++)
			{
				temp_data[temp_row][temp_column] = data[row][column];
				temp_row++;
			}
			temp_column++;
		}
		AnalyzePixel(&tfp, &shift, trigger, total_time, sampling_rate, drive_freq, window, bandpass_filter, filter_bandwidth, temp_data, numRows, averages);

		tfp_data[pixel][0] = tfp;
		shift_data[pixel][0] = shift;
	}
	DisposePtr((Ptr)configPtr);
	delete[] temp_data[0];
	delete[] temp_data;
	delete[] data[0];
	delete[] data;

	/* BEGIN TRANSFER OF DATA TO IGOR HERE.*/
	TransferToIgor(tfpWave, tfp_data);
	TransferToIgor(shiftWave, shift_data);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	delete[] tfp_data[0];
	delete[] tfp_data;
	delete[] shift_data[0];
	delete[] shift_data;

	return 0;
}


/*	RegisterFunction()

Igor calls this at startup time to find the address of the
XFUNCs added by this XOP. See XOP manual regarding "Direct XFUNCs".
*/
static XOPIORecResult
RegisterFunction()
{
	int funcIndex;

	funcIndex = (int)GetXOPItem(0);		// Which function is Igor asking about?
	switch (funcIndex) {
	case 0:						// WAGetWaveInfo(wave)
		return (XOPIORecResult)AnalysisConfig;
		break;
	case 1:						// WAGetWaveInfo(wave)
		return (XOPIORecResult)AnalyzeLine;
		break;
	case 2:
		return (XOPIORecResult)AnalyzeLineOffline;
		break;
	}
	return 0;
}

/*	DoFunction()

Igor calls this when the user invokes one if the XOP's XFUNCs
if we returned NIL for the XFUNC from RegisterFunction. In this
XOP, we always use the direct XFUNC method, so Igor will never call
this function. See XOP manual regarding "Direct XFUNCs".
*/
static int
DoFunction()
{
	int funcIndex;
	void *p;				// Pointer to structure containing function parameters and result.
	int err;

	funcIndex = (int)GetXOPItem(0);	// Which function is being invoked ?
	p = (void*)GetXOPItem(1);		// Get pointer to params/result.
	switch (funcIndex) {
	case 0:
		err = AnalysisConfig((AnalysisConfigParams*)p);
		break;
	case 1:						// WAGetWaveInfo(wave)
		err = AnalyzeLine((AnalyzeLineParams*)p);
		break;
	case 2:						// WAGetWaveInfo(wave)
		err = AnalyzeLineOffline((AnalyzeLineOfflineParams*)p);
		break;
	}
	return(err);
}

/*	XOPEntry()

This is the entry point from the host application to the XOP for all messages after the
INIT message.
*/
extern "C" void
XOPEntry(void)
{
	XOPIORecResult result = 0;

	switch (GetXOPMessage()) {
	case FUNCTION:						// Our external function being invoked ?
		result = DoFunction();
		break;

	case FUNCADDRS:
		result = RegisterFunction();
		break;
	}
	SetXOPResult(result);
}

/*	main(ioRecHandle)

This is the initial entry point at which the host application calls XOP.
The message sent by the host must be INIT.
main() does any necessary initialization and then sets the XOPEntry field of the
ioRecHandle to the address to be called for future messages.
*/
HOST_IMPORT int
main(IORecHandle ioRecHandle)
{
	XOPInit(ioRecHandle);				// Do standard XOP initialization.
	SetXOPEntry(XOPEntry);				// Set entry point for future calls.

	if (igorVersion < 600) {			// Requires Igor Pro 6.00 or later.
		SetXOPResult(OLD_IGOR);			// OLD_IGOR is defined in trEFMAnalysisInterface.h and there are corresponding error strings in trEFMAnalysisInterface.r and trEFMAnalysisInterfaceWinCustom.rc.
		return EXIT_FAILURE;
	}

	SetXOPResult(0L);
	return EXIT_SUCCESS;
}
