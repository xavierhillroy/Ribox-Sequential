#include "LGPEngine.h"
#include "ISA.h"
#include "LGPConfig.h"
#include <random>

void LGPEngine::init_population(){
    // sets all instructions for all programs
    for (int i = 0; i < LGPConfig::TOTAL_INSTRUCTIONS; i ++){
        data.instructions[i] = generate_instruction();
    }

} // V easy GPU transformation 

// Generates instructions
uint32_t LGPEngine::generate_instruction(){
    uint32_t raw_rand = dist_32(rng); // gets random sttream of 32 bits
    return ISA::encode_from_random(raw_rand); // packs and cleans bits (lose some entropy)
}
LGPEngine::LGPEngine():current_generation(0),current_buffer(0), rng(LGPConfig::SEED), dist_32(0,UINT32_MAX),  data(){} // constructor- kept explicitly clean- inits current buffer to 0, sets gen to 0, sets up the seed, and distribution, verbose data init
