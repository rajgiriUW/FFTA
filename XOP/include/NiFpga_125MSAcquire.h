/*
 * Generated with the FPGA Interface C API Generator 13.0.0
 * for NI-RIO 13.0.0 or later.
 */

#ifndef __NiFpga_125MSAcquire_h__
#define __NiFpga_125MSAcquire_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1300
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_125MSAcquire_Bitfile;
 */
#define NiFpga_125MSAcquire_Bitfile "NiFpga_125MSAcquire.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_125MSAcquire_Signature = "989E376BA1809075A27597F5FAD08118";

typedef enum
{
   NiFpga_125MSAcquire_IndicatorBool_edgedetected = 0x8000000E,
} NiFpga_125MSAcquire_IndicatorBool;

typedef enum
{
   NiFpga_125MSAcquire_IndicatorU16_AcquisitionState = 0x8000002E,
} NiFpga_125MSAcquire_IndicatorU16;

typedef enum
{
   NiFpga_125MSAcquire_IndicatorU32_RecordsTransfered = 0x80000014,
} NiFpga_125MSAcquire_IndicatorU32;

typedef enum
{
   NiFpga_125MSAcquire_ControlBool_ExternalTrigger = 0x80000012,
   NiFpga_125MSAcquire_ControlBool_Reset = 0x8000000A,
   NiFpga_125MSAcquire_ControlBool_StartAcq = 0x80000002,
} NiFpga_125MSAcquire_ControlBool;

typedef enum
{
   NiFpga_125MSAcquire_ControlU8_TriggerType = 0x80000026,
} NiFpga_125MSAcquire_ControlU8;

typedef enum
{
   NiFpga_125MSAcquire_ControlI16_TriggerHysteresis = 0x80000022,
   NiFpga_125MSAcquire_ControlI16_TriggerThreshold = 0x8000001E,
} NiFpga_125MSAcquire_ControlI16;

typedef enum
{
   NiFpga_125MSAcquire_ControlU32_NumRecords = 0x80000028,
   NiFpga_125MSAcquire_ControlU32_PreTriggerSamples = 0x80000018,
   NiFpga_125MSAcquire_ControlU32_RecordLength = 0x80000004,
} NiFpga_125MSAcquire_ControlU32;

typedef enum
{
   NiFpga_125MSAcquire_TargetToHostFifoI16_ToHost = 0,
} NiFpga_125MSAcquire_TargetToHostFifoI16;

#endif
