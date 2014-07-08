/*
	trEFMAnalysisPackage.h -- equates for trEFMAnalysisPackage XOP
*/

/* trEFMAnalysisPackage custom error codes */

#define OLD_IGOR 1 + FIRST_XOP_ERR
#define NON_EXISTENT_WAVE 2 + FIRST_XOP_ERR
#define NEEDS_3D_WAVE 3 + FIRST_XOP_ERR

/* Prototypes */
HOST_IMPORT int main(IORecHandle ioRecHandle);

