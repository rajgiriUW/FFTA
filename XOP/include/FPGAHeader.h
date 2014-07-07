#ifndef A_H
#define A_H
#include "NiFpga.h"
#include "XOPStandardHeaders.h"
double** FPGAAcquireData(uint32_t, uint32_t, uint32_t);
int AnalyzePixel(double*, double*, double, double, double, double, int, int, double, double**, int, int);
double** ImportDataFromIgor(waveHndl, int*, int* );
#endif