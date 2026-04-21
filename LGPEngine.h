#ifndef LGP_ENGINE_H
#define LGP_ENGINE_H
#include "LGPConfig.h"
#include <cstdint>
#include <random>
struct PopulationData{
    uint32_t* instructions;
    uint8_t* program_lengths; // assumes programs less then 255
   
    uint32_t* next_gen_instructions; // Write children here during variation
    uint8_t* next_gen_lengths;       // Child lengths
    
    float* fitness_scores;
};
class LGPEnvironment{
    private: 
        PopulationData data;
        int current_generation;
        uint32_t* current_instructions; 
        uint32_t* current_program_lengths;
	    std::mt19937 rng; 
    public: 
        LGPEnvironment();
        ~LGPEnvironment();
       	void init(uint_32_t seed);
	void mutate();
        void crossover();
        int tournament_selection(); // returns the program index that is selected
        void vary();// crossover + mutation 
        void evolve(); // evolutionary loop 


};


    

#endif //LGP_ENGINE_H
