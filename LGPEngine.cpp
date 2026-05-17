#include "LGPEngine.h"
#include "ISA.h"
#include "LGPConfig.h"
#include <random>
#include "Evaluator.h"
#include "Dataset.h"
#include <cassert>
#include <iostream>
#include <cstring>
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
// constructor- kept explicitly clean- inits current buffer to 0, sets gen to 0, sets up the seed, and distribution, verbose data init
LGPEngine::LGPEngine()
    : current_generation(0),
    current_buffer(0), 
    rng(LGPConfig::SEED), 
    dist_32(0,UINT32_MAX), dist_pop(0,LGPConfig::POPULATION_SIZE-1),
    dist_unit(0.0f, 1.0f),
    dist_field(0, 4),
    data()
{} 


// This is program view, a method that returns the program view structure, contains length of cur prog and pointer to the start of it 
ProgramView LGPEngine::view_program(int i) const{
    // because we ping pong betweeen 2 buffers between generations, we must select the live buffer

    const std::vector<uint32_t>& instr = (current_buffer ==0)
    ? data.instructions
    : data.next_gen_instructions;

    // selects current live buffer for length of program
    const std::vector<uint8_t>& lens = (current_buffer == 0)
    ? data.program_lengths
    : data.next_gen_lengths;
    
    return ProgramView{
        instr.data() + i * LGPConfig::MAX_PROGRAM_SIZE, // here we are getting the raw pointer of instr vector (start) and getting the adress of the prog we are working withh
        static_cast<int>(lens[i]) // explicitly casting to int (widening from uint 8)
    };
}
void LGPEngine::evaluate_all(const Dataset& dataset){
    for (int agent = 0; agent < LGPConfig::POPULATION_SIZE; ++agent){
        const ProgramView prog = view_program(agent);
        data.fitness_scores[agent] = Fitness::mse_to_fitness(Evaluator::evaluate_sr_mse(prog, dataset));
    }
}
float Fitness::mse_to_fitness(float mse){ // converts lower is better mse to higher is better for fitness selection
    if (!std::isfinite(mse)) return 0.0f;
    return 1.0f / (1.0f + mse);

}
int LGPEngine::tournament_selection(){
    // select k agents, return the one with the highest fitness 
    //reads from the current buffer fitness fitness_scores

    int best_idx = dist_pop(rng); // get ran agent in population 
    float best_fit = data.fitness_scores[best_idx]; // gets their fitness score 

    // make sure that fitness score is not NAN shouldnt be if evaluate all ran before xx 
    assert(!std::isnan(best_fit) && "Selection called before evaluate all (fitness NAN)");

    for (int agent = 1; agent < LGPConfig::TOURNAMENT_SIZE; ++agent){
        int cand = dist_pop(rng); // get ran agent in population 
        float cand_fit = data.fitness_scores[cand]; // gets their fitness score 
        assert(!std::isnan(cand_fit) && "Selection called before evaluate all (fitness NAN)");

        if (cand_fit > best_fit){
            //std::cout<<"Candidate fitness ("<<cand_fit<<") > Best fit ("<<best_fit<<")"<<std::endl;
            best_fit = cand_fit;
            best_idx = cand;
        }
    }
    return best_idx;
}
// LGPEngine.cpp
#include <cstring>  // memcpy

void LGPEngine::crossover(const ProgramView& a, const ProgramView& b,
                          uint32_t* child_a, uint8_t& child_a_len,
                          uint32_t* child_b, uint8_t& child_b_len) {
    // Cut point in [1, min(len_a, len_b)). Both children get at least one
    // instruction from each parent. If either parent is length 1, we can't
    // do a meaningful crossover -- just clone both parents.
    const int min_len = std::min(a.length, b.length);
    if (min_len < 2) {
        std::memcpy(child_a, a.instructions, a.length * sizeof(uint32_t));
        std::memcpy(child_b, b.instructions, b.length * sizeof(uint32_t));
        child_a_len = static_cast<uint8_t>(a.length);
        child_b_len = static_cast<uint8_t>(b.length);
        return;
    }

    std::uniform_int_distribution<int> dist_cut(1, min_len - 1);
    const int cut = dist_cut(rng);

    // Child A: a's prefix + b's suffix. Final length = b.length.
    std::memcpy(child_a, a.instructions, cut * sizeof(uint32_t));
    std::memcpy(child_a + cut, b.instructions + cut,
                (b.length - cut) * sizeof(uint32_t));
    child_a_len = static_cast<uint8_t>(b.length);

    // Child B: b's prefix + a's suffix. Final length = a.length.
    std::memcpy(child_b, b.instructions, cut * sizeof(uint32_t));
    std::memcpy(child_b + cut, a.instructions + cut,
                (a.length - cut) * sizeof(uint32_t));
    child_b_len = static_cast<uint8_t>(a.length);
}

// this metthod could cause divergence on GPU 
void LGPEngine::mutate(uint32_t* program, uint8_t& length) {
    if (length == 0) return;

    if (dist_unit(rng) < LGPConfig::MICRO_RATE) {
        std::uniform_int_distribution<int> dist_inst(0, length - 1);
        const int idx = dist_inst(rng);
        program[idx] = micro_mutate_instruction(program[idx]);
    }

    if (dist_unit(rng) < LGPConfig::INSERT_TAIL_RATE) {
        if (length < LGPConfig::MAX_PROGRAM_SIZE) {
            program[length] = generate_instruction();
            ++length;
        }
    }

    if (dist_unit(rng) < LGPConfig::DELETE_TAIL_RATE) {
        if (length > 1) {
            --length;
        }
    }
}
uint32_t LGPEngine::micro_mutate_instruction(uint32_t instr) {
    // Pick one of 5 fields to perturb. Uniform over fields, not bits --
    // gives op-changes and mode-flips comparable mutation pressure even
    // though they occupy different bit widths.
    const int field = dist_field(rng);  // [0, 4]
    const uint32_t r = dist_32(rng);    // source of random bits

    switch (field) {
        // general formula is we are creating a mask and shifting it into its appropriate spot on the ISA (32 bits)
        case 0: {  // op
            const uint32_t mask = LGPConfig::OPERATION_MASK << ISA::OP_SHIFT; 
            const uint32_t new_bits = (r & LGPConfig::OPERATION_MASK) << ISA::OP_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 1: {  // dest
            const uint32_t mask = LGPConfig::REGISTER_MASK << ISA::DEST_SHIFT;
            const uint32_t new_bits = (r & LGPConfig::REGISTER_MASK) << ISA::DEST_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 2: {  // src1
            const uint32_t mask = LGPConfig::REGISTER_MASK << ISA::SRC1_SHIFT;
            const uint32_t new_bits = (r & LGPConfig::REGISTER_MASK) << ISA::SRC1_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 3: {  // src2 index (preserves the mode bit above it)
            const uint32_t mask = ISA::SRC2_INDEX_MASK << ISA::SRC2_SHIFT;
            const uint32_t new_bits = (r & ISA::SRC2_INDEX_MASK) << ISA::SRC2_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 4: {  // src2 mode bit (toggle register/constant)
            return instr ^ (static_cast<uint32_t>(ISA::MASK_MODE_BIT) << ISA::SRC2_SHIFT);
        }
    }
    return instr;  // unreachable
}