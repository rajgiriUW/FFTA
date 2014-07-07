#include "NiFpga.h"
#include "NiFpga_125MSAcquire.h"
#include "XOPStandardHeaders.h"	
#include <math.h>

static int gCallSpinProcess = 1;

double** FPGAAcquireData(uint32_t numRecords, uint32_t recordLength, uint32_t preTriggerSamples)
{
	NiFpga_Session session;
	NiFpga_Status status;
	NiFpga_Bool						edgeDetected;
	double**						data;
	uint32_t						recordsTransfered;
	int16_t*						signal;
	char							noticeString[50];
	int result;
	double scaleFactor = pow(2.0,  15.0);
	// initialize storage pointer
	const int nrow = (int) recordLength, ncol = (int)numRecords, nelem = nrow*ncol;
	data = new double*[nrow];
	data[0] = new double[nelem];
	for (int i = 1; i < nrow; i++)
	{
		data[i] = data[i - 1] + ncol;
	}
	int16_t *s = new int16_t[recordLength];

	status = NiFpga_Initialize();
	// Open the Bitfile located in the IGOR Extensions Folder, where this XOP will be placed
	if (NiFpga_IsNotError(status))
	{
		NiFpga_MergeStatus(&status, NiFpga_Open("C:\\Program Files (x86)\\WaveMetrics\\Igor Pro Folder\\Igor Extensions\\"NiFpga_125MSAcquire_Bitfile,
			NiFpga_125MSAcquire_Signature,
			"RIO0",
			0,
			&session));
		sprintf(noticeString, "Error in opening bitfile:%i\r", status);
		XOPNotice(noticeString);
	}

	if (NiFpga_IsNotError(status))
	{
		// Write all the requested parameters to the FPGA Acquisition System.

		NiFpga_MergeStatus(&status,
			NiFpga_WriteU32(session,
			NiFpga_125MSAcquire_ControlU32_NumRecords,  // Averages
			numRecords));

		NiFpga_MergeStatus(&status,
			NiFpga_WriteU32(session,
			NiFpga_125MSAcquire_ControlU32_RecordLength, // Points per Average
			recordLength));

		NiFpga_MergeStatus(&status,
			NiFpga_WriteU32(session,
			NiFpga_125MSAcquire_ControlU32_PreTriggerSamples, // Number of pretrigger samples
			preTriggerSamples));

		NiFpga_MergeStatus(&status,
			NiFpga_WriteBool(session,
			NiFpga_125MSAcquire_ControlBool_ExternalTrigger, // Tells the FPGA to use an external trigger instead of the built-in software trigger
			1));

		// Start the Acquisiton
		NiFpga_MergeStatus(&status,
			NiFpga_WriteBool(session,
			NiFpga_125MSAcquire_ControlBool_StartAcq,
			1));
	}
	if (NiFpga_IsNotError(status))
	{
		NiFpga_MergeStatus(&status,
			NiFpga_ReadU32(session,
			NiFpga_125MSAcquire_IndicatorU32_RecordsTransfered,
			&recordsTransfered));
		// Keep recording until we have reached the desired amount of triggers
		while (recordsTransfered < numRecords - 1)
		{
			if (gCallSpinProcess && SpinProcess()) {		// Spins cursor and allows background processing.
				result = -1;								// User aborted.
				break;
			}

			NiFpga_MergeStatus(&status,
				NiFpga_ReadU32(session,
				NiFpga_125MSAcquire_IndicatorU32_RecordsTransfered,
				&recordsTransfered));

			// Wait until we trigger.
			NiFpga_MergeStatus(&status,
				NiFpga_ReadBool(session,
				NiFpga_125MSAcquire_IndicatorBool_edgedetected,
				&edgeDetected));
			while (edgeDetected == 0)
			{
				NiFpga_MergeStatus(&status,
					NiFpga_ReadBool(session,
					NiFpga_125MSAcquire_IndicatorBool_edgedetected,
					&edgeDetected));
			}


			// Transfer data from the FPGA FIFO to a temporary storage buffer
			NiFpga_MergeStatus(&status,
				NiFpga_ReadFifoI16(session,
				NiFpga_125MSAcquire_TargetToHostFifoI16_ToHost,
				s,
				recordLength,
				NiFpga_InfiniteTimeout,
				NULL));

			// Write from the temp buffer into the data transfer buffer
			for (int i = 0; i < recordLength; i++)
			{
				data[i][recordsTransfered] = (double)s[i]/scaleFactor;
			}

		}
		/* Reset the FPGA back to the "Wait for Start Trigger" State*/
		NiFpga_MergeStatus(&status,
			NiFpga_WriteBool(session,
			NiFpga_125MSAcquire_ControlBool_StartAcq,
			0));
		NiFpga_MergeStatus(&status,
			NiFpga_WriteBool(session,
			NiFpga_125MSAcquire_ControlBool_Reset,
			1));

		NiFpga_MergeStatus(&status,
			NiFpga_WriteBool(session,
			NiFpga_125MSAcquire_ControlBool_Reset,
			0));
		NiFpga_MergeStatus(&status, NiFpga_Close(session, 0));
	}

	NiFpga_MergeStatus(&status, NiFpga_Finalize());

	delete[] s;
	return data;
}