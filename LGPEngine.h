#ifndef LGP_ENGINE_H
#define LGP_ENGINE_H
#include "LGPConfig.h"
#include <cstdint>
#include <random>
#include <vector>
#include <limits>
#include "Evaluator.h"
#include "Dataset.h"

// =============================================================================
// PopulationData: owns all the buffers that describe the population.
//
// This struct is deliberately just data + allocation -- it doesn't implement
// any of the evolutionary loop. LGPEngine operates on it. Keeping the two
// separate means (a) PopulationData can later be swapped for a GPU-backed
// equivalent (same fields, but storage is cudaMalloc'd) without touching
// engine logic, and (b) tests can construct a PopulationData in isolation.
//
// Memory layout recap (see LGPConfig.h for the "why"):
//   - instructions[]:    flat buffer of TOTAL_INSTRUCTIONS uint32_t words.
//                        Program i occupies the slice
//                          [i * MAX_PROGRAM_SIZE, i * MAX_PROGRAM_SIZE + program_lengths[i])
//                        Slots past program_lengths[i] exist but aren't
//                        executed.
//   - program_lengths[]: one byte per program; gates how far the interpreter
//                        walks into that program's slice.
//
// Two parallel "next-gen" buffers implement ping-pong double-buffering for
// variation: children are written into next_gen_* while the current
// generation remains readable. After variation completes, the engine flips
// a pointer/flag (current_buffer) to swap roles rather than copying. This
// avoids allocation churn per generation and mirrors how the GPU version
// will want to work.
// =============================================================================

// Container that contains population information
struct PopulationData {

    // ---- Current generation ------------------------------------------------
    std::vector<uint32_t> instructions;      // Flat buffer of all programs'
                                             // instructions; size = TOTAL_INSTRUCTIONS.
    std::vector<uint8_t>  program_lengths;   // Active length of each program.
                                             // uint8_t because MAX_PROGRAM_SIZE fits
                                             // comfortably under 255.
    static_assert(LGPConfig::MAX_PROGRAM_SIZE <= 255, "MAX PROGRAM SIZE LARGER THEN uint8 can store");


    // ---- Next generation (children go here during variation) ---------------
    std::vector<uint32_t> next_gen_instructions;  // Write children here during variation.
    std::vector<uint8_t>  next_gen_lengths;       // Child lengths.

    // ---- Fitness ----------------------------------------------------------
    std::vector<float>    fitness_scores;    // One per program; filled by the
                                             // evaluator.

    // -------------------------------------------------------------------------
    // Default constructor sizes all buffers from the compile-time constants
    // in LGPConfig, so the struct is always ready-to-use after construction.
    // No manual resize() calls at the use site.
    //
    // Initialization values chosen deliberately:
    //   - instructions / next_gen_instructions -> 0: zero-filled instruction
    //     slots decode to a valid (if meaningless) instruction, so a missed
    //     init() can't produce UB in the interpreter.
    //   - program_lengths -> STARTING_PROGRAM_SIZE: same defensive reasoning.
    //     A missed init() still leaves every program with a valid length.
    //   - next_gen_lengths -> 0: children don't exist yet; variation fills
    //     these in.
    //   - fitness_scores -> NaN: sentinel for "not yet evaluated". Any
    //     comparison with NaN returns false, so selection run on an
    //     unevaluated population fails visibly rather than silently picking
    //     the zero-fitness winners.
    // -------------------------------------------------------------------------
    PopulationData()
        : instructions(LGPConfig::TOTAL_INSTRUCTIONS, 0),
          program_lengths(LGPConfig::POPULATION_SIZE,    LGPConfig::STARTING_PROGRAM_SIZE),
          next_gen_instructions(LGPConfig::TOTAL_INSTRUCTIONS, 0),
          next_gen_lengths(LGPConfig::POPULATION_SIZE,    0),
          fitness_scores(LGPConfig::POPULATION_SIZE,
                        std::numeric_limits<float>::quiet_NaN())
    {}
};

// =============================================================================
// LGPEngine: owns the evolutionary loop.
//
// Holds the population data, the RNG, and the generation counter, and
// exposes the operators (init / mutate / crossover / select / vary / evolve)
// that drive the loop. All randomness flows through the engine's single
// mt19937 so that runs are reproducible given LGPConfig::SEED.
// =============================================================================
class LGPEngine {
private:
    // ---- Generation state --------------------------------------------------
    int current_generation;                        // Counter; starts at 0 in ctor.
    int current_buffer;                            // 0 or 1: indicates which of
                                                   // the two buffers in PopulationData
                                                   // is the "live" generation.
                                                   // Flipped after each variation pass.

    // ---- RNG ---------------------------------------------------------------
    std::mt19937 rng;      // Seeded with LGPConfig::SEED
                                                        // in the constructor.
    std::uniform_int_distribution<uint32_t>   dist_32;  // Uniform over the full
                                                        // uint32_t range; used to
                                                        // generate random instruction
                                                        // words.
    //Population distribution, random over 0 -> popsize - 1
    std::uniform_int_distribution<int> dist_pop;

    // Members to add:
    std::uniform_real_distribution<float> dist_unit;       // [0, 1)
    std::uniform_int_distribution<int> dist_field;      // [0, 4]

    // ---- Population --------------------------------------------------------
    PopulationData data;                           // All buffers; default-constructed.

public:

    LGPEngine();
    ~LGPEngine() = default;
    ProgramView view_program(int i) const; // returns a program view object ( cur instruction part and cur length)
    void evaluate_all(const Dataset& dataset); // evluates entire population... this loop will disappear on GPU - be assigned to diff warps 

    // ---- Public evolutionary interface -------------------------------------
    void init_population();                  // Randomize the initial population.
    void mutate(uint32_t* program, uint8_t& length);
    void crossover(const ProgramView& a, const ProgramView& b,
                          uint32_t* child_a, uint8_t& child_a_len,
                          uint32_t* child_b, uint8_t& child_b_len);
    int  tournament_selection();  // Returns the program index that is selected.
    void vary();                  // Crossover + mutation over the whole population.
    void evolve();                // Top-level evolutionary loop.

    const PopulationData& get_data() const { return data; } // access data for testing 
    PopulationData& get_mutable_data() { return data; } // access data for testing 
    uint32_t micro_mutate_instruction(uint32_t instruction);


private:
    // ---- Internal helpers --------------------------------------------------
    uint32_t generate_instruction();  // One random, encoding-valid instruction word.
    

};
namespace Fitness {
    float mse_to_fitness(float mse);
}
#endif  // LGP_ENGINE_H