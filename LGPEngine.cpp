#include "LGPEngine.h"
#include "ISA.h"
#include "LGPConfig.h"
#include <random>
#include "Evaluator.h"
#include "Dataset.h"
#include <cassert>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
void LGPEngine::init_population(){
    // sets all instructions for all programs
    for (int i = 0; i < LGPConfig::TOTAL_INSTRUCTIONS; i ++){
        cur_instructions_mutable()[i] = generate_instruction();
       
    }
       // Set initial program lengths. Using std::fill for clarity vs vector::assign.
    std::fill(cur_lengths_mutable().begin(),
              cur_lengths_mutable().end(),
              static_cast<uint8_t>(LGPConfig::STARTING_PROGRAM_SIZE));

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
    // because we ping pong betweeen 2 buffers between generations - live bufer selection done in accessor 
    
    return ProgramView{
        cur_instructions().data() + i * LGPConfig::MAX_PROGRAM_SIZE, // here we are getting the raw pointer of instr vector (start) and getting the adress of the prog we are working withh
        static_cast<int>(cur_lengths()[i]) // explicitly casting to int (widening from uint 8)
    };
}
void LGPEngine::evaluate_all(const Dataset& dataset){
    for (int agent = 0; agent < LGPConfig::POPULATION_SIZE; ++agent){
        if(std::isnan(cur_fitness()[agent])){
            const ProgramView prog = view_program(agent);
            cur_fitness_mutable()[agent] = Fitness::mse_to_fitness(Evaluator::evaluate_sr_mse(prog, dataset));
        }
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
    float best_fit = cur_fitness()[best_idx]; // gets their fitness score 

    // make sure that fitness score is not NAN shouldnt be if evaluate all ran before xx 
    assert(!std::isnan(best_fit) && "Selection called before evaluate all (fitness NAN)");

    for (int agent = 1; agent < LGPConfig::TOURNAMENT_SIZE; ++agent){
        int cand = dist_pop(rng); // get ran agent in population 
        float cand_fit = cur_fitness()[cand]; // gets their fitness score 
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

    // if (dist_unit(rng) > LGPConfig::CROSSOVER_RATE) {return;}
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
// Doesnt mirror crossover perfectly cuz probalistic natture baked into method... oops 
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
void LGPEngine::vary_pair(int dstA, int dstB){
    // get parents 
    int pA_idx = tournament_selection();
    int pB_idx = tournament_selection();
    ProgramView pA = view_program(pA_idx);
    ProgramView pB = view_program(pB_idx);

    // buffers chidren
    uint32_t* childA_buf = next_gen_ptr(dstA);
    uint32_t* childB_buf = next_gen_ptr(dstB);
    uint8_t& childA_length = next_lengths()[dstA];
    uint8_t& childB_length = next_lengths()[dstB];

    

    // if we dont crossover
    if (!(dist_unit(rng) < LGPConfig::CROSSOVER_RATE)){
    //clone parents 
        std::memcpy(childA_buf, pA.instructions, pA.length * sizeof(uint32_t) );// copy it exactly, destination, source, size in bytes
        std::memcpy(childB_buf, pB.instructions, pB.length * sizeof(uint32_t));

        childA_length = pA.length;
        childB_length = pB.length;
    }
    else{
        crossover(pA, pB, childA_buf, childA_length, childB_buf, childB_length);
    }
    // now do the mutation
    mutate(childA_buf, childA_length);
    mutate(childB_buf, childB_length);

    // set their fitness as Nan 
    next_fitness()[dstA] = std::numeric_limits<float>::quiet_NaN();
    next_fitness()[dstB] = std::numeric_limits<float>::quiet_NaN();
}
std::vector<int> LGPEngine::top_k_indices(int k) const{
    // create vector of popsize and fill with numfrom 0 to popsize - 1
    std::vector<int> idx(LGPConfig::POPULATION_SIZE);
    std::iota(idx.begin(), idx.end(), 0);

    const auto& fit = cur_fitness(); // get reference to cur fitness vectors 

    // sort by top k indices 
    // get iterator for being, sorts up to iterator + k, sort partial from it begin to it end. 
    // lambda as comparator to sort indices by their fitness
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&fit](int a, int b){return fit[a]> fit[b];}); // uses lambda to sort indices by fitness 
    idx.resize(k); // resize to k elements idx of top k fittest individuals 
    return idx;

}
// WORTH CONSIDERING IN GPU PORT - this COPY Program copies dif amount based on length woould 
void LGPEngine::copy_elite_to_next(int srcIdx, int dstIdx){
    const uint32_t* source = cur_instructions().data() + srcIdx * LGPConfig::MAX_PROGRAM_SIZE; // ptr to start of program at srcIdx  in current buffer (pointer to program at s)
    uint32_t* dest = next_gen_ptr(dstIdx); // pointer to program at destination index in next gen buffer 
    uint8_t length = cur_lengths()[srcIdx]; // lenght of program we are copying 

    // now the memcopy 
    std::memcpy(dest, source, sizeof(uint32_t) * length);
    next_lengths()[dstIdx] = length;
    next_fitness()[dstIdx] = cur_fitness()[srcIdx];
}

void LGPEngine::vary(){
    // fill elites - carries over length and fitness and all that good stuff 
    std::vector<int> topK = top_k_indices(LGPConfig::ELITE_COUNT);

    for (int e = 0; e < LGPConfig::ELITE_COUNT; ++e){
        copy_elite_to_next(topK[e], e);
    }
    // populate with real variation 
    for (int children = LGPConfig::ELITE_COUNT; children < LGPConfig::POPULATION_SIZE; children +=2){
        vary_pair(children, children +1); // passing in destination indices 
    }
    flip_generation();
}