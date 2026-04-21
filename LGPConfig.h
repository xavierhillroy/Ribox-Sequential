#ifndef LGP_CONFIG_H
#define LGP_CONFIG_H

#include <cstdint>
// THESE ARE ALL COMPILE TIME COMMANDS

namespace LGPConfig{
    // LOG 2 to calculate bits needed 
    constexpr int log2(int n){
        return (n<=1) ? 0: 1 + log2(n/2);
    }
    constexpr int MAX_GENERATIONS = 100;
    constexpr int NUM_PROGRAMS = 100; 

    // MEMORY LAYOUT - We are storing all programs as a list of uint_32 instructions (just one big list of instructions, programs seperated by indices)  
    constexpr int MAX_PROGRAM_SIZE = 16; // Max num instructs prog can have 
    constexpr int POPULATION_SIZE = 2;
    constexpr int TOTAL_INSTRUCTIONS = POPULATION_SIZE * MAX_PROGRAM_SIZE;
    constexpr int STARTING_PROGRAM_SIZE = 8

    constexpr int NUM_CONSTANTS = 8; 
    constexpr int CONSTANT_BITS = log2(NUM_CONSTANTS);
    constexpr int CONSTANT_MASK = NUM_CONSTANTS - 1;
    constexpr std::array<float, NUM_CONSTANTS> CONSTANTS = {
        0.0f, 1.0f, -1.0f, 3.14159f, 0.5f, 2.0f, 10.0f, 100.0f
    };

    constexpr int NUM_REGISTERS = 8; // MUST BE A POWER OF 2! (max 2^7 )
    constexpr int REGISTER_BITS = log2(NUM_REGISTERS);
    constexpr int REGISTER_MASK = NUM_REGISTERS - 1;
    
   constexpr int NUM_OPERATIONS = 8; // mAx 2^8 
   constexpr int OPERATIONS_BITS = log2(NUM_OPERATIONS); // Num active bits cause we have max 8 bits for this but often we wont have 256 ops 
   constexpr int OPERATION_MASK = NUM_OPERATIONS - 1;  // Mask based on num ops - for safety. Prevents us op vals that exceed our num of ops. 
   // Ex we have 8 bits and lets say its 1111 1111-  256 ops available. lets say we only have 8  operatiosn we do a mask of the num 7 (111) when we and to be 0000 0111 

   // RATES 
   constexpr float insert_end = 0.8f; 
   constexpr float pop_swap = 0.75f;
   constexpr float replace = 0.8f;
   constexpr float micro = 0.8f; 

   constexpr float crossover = 0.9f; 
   constexpr float ELITES = 0.6f;


}
#endif
