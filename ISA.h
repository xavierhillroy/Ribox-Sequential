#ifndef ISA_H
#define ISA_H
#include <cstdint>
#include "LGPConfig.h"

namespace ISA {
// Order: [SRC2 (8)] [SRC1 (8)] [DEST (8)] [OP (8)] - LITTLE ENDIAN STYLE STORAGE 
constexpr int OP_SHIFT = 0; 
constexpr int DEST_SHIFT = 8;
constexpr int SRC1_SHIFT = 16;
constexpr int SRC2_SHIFT = 24;

// OPERATIONS AVAILABLE 
enum OpCode : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    SIN = 4,
    COS = 5,
    LT = 6,
    GT = 7,



    OPCOUNT = NUM_OPERATIONS


};
// SPECIFIC MASK FOR THE SPLIT BYTE at the end- SRC 2 has the mode, 0 for reg and 1 for constant as its 7th bit 
constexpr uint8_t MASK_MODE_BIT = 0x80; // Binary 1000 0000
// The bottom 7 bits are the raw index
constexpr uint8_t MASK_RAW_INDEX = 0x7F; // Binary 0111 1111
}
#endif // ISA 
