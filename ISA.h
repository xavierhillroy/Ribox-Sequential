#ifndef ISA_H
#define ISA_H
#include <cstdint>
#include "LGPConfig.h"
//
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
constexpr uint8_t MASK_MODE_BIT = 0x80; // Binary 1000 0000 /
// The bottom 7 bits are the raw index
constexpr uint8_t MASK_RAW_INDEX = 0x7F; // Binary 0111 1111 (either the index of reg or constant)

// ENCODING AND DECODING FOR EASE OF USE 
 
inline uint32_t encode_from_random(uint32_t raw_rand){
    /**
    Example of what we are doing 
     encoded_instruct = 0000 0000 0000 0000 0000 0000 0000 0000 
     raw_rand = 0010 0111 1111 0010 1010 0011 1110 1001
     let us say we are encoding the reg (dest) portion at this poitn and lets say its 3 bits (8 registers - must use 2^n registers) then the shift mask is 7 (111)cuz 3 1 bits
     dest shift = 8 
     REGISTER_MASK = 0000 0000 0000 0000 0000 0000 0000 0111
     then we shift it by 8 to left 0000 0000 0000 0000 0000 0000 0000 0111 << 8 =  0000 0000 0000 0000 0000 0111 0000 0000
     Now we & raw_rand & shiftted reg MASk:  0010 0111 1111 0010 1010 0011 1110 1001 & 0000 0000 0000 0000 0000 0111 0000 0000 = 0000 0000 0000 0000 0000 `0011` 0000 0000 - leaving only these bits impacted (the ones impacting the dest reg)
     Then we or that with the encoded_instruct stream thus cleaning the uneeded garbage bits ( we could have left it but for legibility we cleaned it )
     */
    uint32_t encoded_instruct = 0; 
   

    encoded_instruct |= raw_rand & (LGPConfig::OPERATION_MASK << OP_SHIFT); // Compactting Op
    encoded_instruct |= raw_rand & (LGPConfig::REGISTER_MASK << DEST_SHIFT); // COMPACTING DEST
    encoded_instruct |= raw_rand & (LGPConfig::REGISTER_MASK << SRC1_SHIFT); // COmpacting SRC1
    // SRC 2 is special because we need to preserve the last bit because that is what we use for encoding the flag 
    encoded_instruct |= raw_rand & (LGPConfig::REGISTER_MASK << SRC2_SHIFT); // this compacts the index part but will remove the flag bit 
    encoded_instruct |= raw_rand & (MASK_MODE_BIT <<SRC2_SHIFT); /// now we added the flag back 
    return encoded_instruct;
}

inline uint32_t encode_manual(uint8_t op, uint8_t dest, uint8_t src1, uint8_t src2){
    uint32_t encoded_instruct = 0; 
    encoded_instruct |= (static_cast<uint32_t>(op & LGPConfig::OPERATION_MASK  )<< OP_SHIFT );
    encoded_instruct |= (static_cast<uint32_t>(dest & LGPConfig::REGISTER_MASK  )<< DEST_SHIFT );
    encoded_instruct |= (static_cast<uint32_t>(src1 & LGPConfig::REGISTER_MASK  )<< SRC1_SHIFT );

    // src 2 has 2 parts, 
     encoded_instruct |= (static_cast<uint32_t>(src2 & LGPConfig::REGISTER_MASK  )<< SRC2_SHIFT ); // compacts tthe index partt
     encoded_instruct |= (static_cast<uint32_t>(src2 & MASK_MODE_BIT  )<< SRC2_SHIFT ); // adds flag

     return encoded_instruct;
}

// DECODING NOW 
}
#endif // ISA 
