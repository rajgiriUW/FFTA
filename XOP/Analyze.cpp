#include "XOPStandardHeaders.h"
#include "Python.h"
#include "C:\Python27\Lib\site-packages\numpy\core\include\numpy\arrayobject.h"
#include <chrono>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION // we aren't using enough numpy API to worry about this.

 int AnalyzePixel(double *tfp, double* shift, double trigger, double total_time, double sampling_rate, double drive_freq, int window, int bandpass_filter, double filter_bandwidth, double** signal_array, int n_points, int n_signals)
{
	 PyObject *pName, *pModule, *pDict, *pClass, *pInstance, *pValue, *classArgs, *mat, *pyTfp, *pyShift, *pValue2, *pFunc;
	 PyObject *pyTrigger, *pyTotal_Time, *pySampling_Rate, *pyDrive_Freq, *pyWindow, *pyBandpass_Filter, *pyFilter_Bandwidth;

	 char noticeStr[50]; // string for XOP messaging

	 Py_Initialize();

	 //Every Numpy C API needs this call.
	 import_array1(-1);

	 //name of the python script to interface with.
	 pName = PyString_FromString("pixel");

	 // Import the analysis code.
	 pModule = PyImport_Import(pName);

	 if (pModule == NULL){
		 PyErr_Print();
	 }

	 // Get the callable objects.
	 pDict = PyModule_GetDict(pModule);

	 // Define values for parameter dictionary.
	 pyTrigger = PyFloat_FromDouble(trigger);
	 pyTotal_Time = PyFloat_FromDouble(total_time);
	 pySampling_Rate = PyFloat_FromDouble(sampling_rate);
	 pyDrive_Freq = PyFloat_FromDouble(drive_freq);
	 pyWindow = PyBool_FromLong(window);
	 pyBandpass_Filter = PyBool_FromLong(bandpass_filter);
	 pyFilter_Bandwidth = PyFloat_FromDouble(filter_bandwidth);

	 // Place some key:values into our parameter dictionary.
	 classArgs = PyDict_New();
	 PyDict_SetItemString(classArgs, "trigger", pyTrigger);
	 PyDict_SetItemString(classArgs, "total_time", pyTotal_Time);
	 PyDict_SetItemString(classArgs, "sampling_rate", pySampling_Rate);
	 PyDict_SetItemString(classArgs, "drive_freq", pyDrive_Freq);
	 PyDict_SetItemString(classArgs, "window", pyWindow);
	 PyDict_SetItemString(classArgs, "bandpass_filter", pyBandpass_Filter);
	 PyDict_SetItemString(classArgs, "filter_bandwidth", pyFilter_Bandwidth);

	 // Build the name of a callable class .
	 pClass = PyDict_GetItemString(pDict, "Pixel");

	 // dimension of the signal_array.
	 int mdim[] = { n_points, n_signals };

	 // Send our data into a numpy ndarray.
	 mat = PyArray_SimpleNewFromData(2, mdim, PyArray_DOUBLE, signal_array[0]);

	 // Create tuple of arguments used to initialize the class.
	 PyObject* args = PyTuple_New(2);
	 PyTuple_SetItem(args, 0, mat);
	 PyTuple_SetItem(args, 1, classArgs);

	 // Initialize the class into pInstance.
	 pInstance = PyObject_CallObject(pClass, args);

	 //if (pInstance == NULL){
	 //}

	 // Call a method of the class with no parameters

	 pValue = PyObject_CallMethod(pInstance, "get_tfp", NULL);
	 Py_XDECREF(pInstance);
	 Py_XDECREF(mat);
	 Py_XDECREF(classArgs);

	 if (pValue == NULL){
	 }

	 if (pValue != NULL)

	 {
		 pyTfp = PyTuple_GetItem(pValue, 0);
		 pyShift = PyTuple_GetItem(pValue, 1);

		 (*tfp) = PyFloat_AsDouble(pyTfp);
		 (*shift) = PyFloat_AsDouble(pyShift);
		 Py_XDECREF(pValue);
	 }
	 else
	 {
		 PyErr_Print();
	 }
	 // Clean up
	 Py_XDECREF(pModule);
	 Py_XDECREF(pName);
	 //



	 //Py_Finalize();
	 return 0;
 }
