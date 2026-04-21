#include "LGPEngine.h"
#include "LGPConfig.h"
#include "ISA.h"
#include <iostream>
#include <cassert>
#include <cmath>

void print_instruction(uint32_t instr) {
    std::cout << "op=" << (int)ISA::get_op(instr)
              << " dest=r" << (int)ISA::get_dest_index(instr)
              << " src1=r" << (int)ISA::get_src1_index(instr)
              << " src2=" << (ISA::is_src2_constant(instr) ? "C[" : "r[")
              << (int)ISA::get_src2_index(instr) << "]"
              << "\n";
}

int main() {
    LGPEngine engine;
    engine.init_population();

    const PopulationData& data = engine.get_data();

    // Structural invariants
    assert(data.instructions.size()    == LGPConfig::TOTAL_INSTRUCTIONS);
    assert(data.program_lengths.size() == LGPConfig::POPULATION_SIZE);
    assert(data.fitness_scores.size()  == LGPConfig::POPULATION_SIZE);

    // Every program should have the starting length
    for (auto len : data.program_lengths) {
        assert(len == LGPConfig::STARTING_PROGRAM_SIZE);
    }

    // Fitness should still be NaN sentinel (we haven't evaluated)
    for (auto f : data.fitness_scores) {
        assert(std::isnan(f));
    }

    // Every decoded instruction in the active range should have valid fields
    for (int p = 0; p < LGPConfig::POPULATION_SIZE; ++p) {
        int base = p * LGPConfig::MAX_PROGRAM_SIZE;

        for (int i = 0; i < data.program_lengths[p]; ++i) {
            uint32_t instr = data.instructions[base + i]; // to access each component 
            assert(ISA::get_op(instr)         < LGPConfig::NUM_OPERATIONS);
            assert(ISA::get_dest_index(instr) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src1_index(instr) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src2_index(instr) < LGPConfig::NUM_REGISTERS);
        }
    }

    // Eyeball test: print program 0
    std::cout << "Program 0:\n";
    int base = 0;
    for (int i = 0; i < data.program_lengths[0]; ++i) {
        std::cout << "  [" << i << "] ";
        print_instruction(data.instructions[base + i]);
    }

    std::cout << "All assertions passed.\n";
    return 0;
}