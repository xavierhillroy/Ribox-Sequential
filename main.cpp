#include "LGPEngine.h"
#include "Dataset.h"
#include "LGPConfig.h"
#include <iostream>

// Identity target f(x) = x -- the trivial sanity problem. A 0-instruction
// program already solves it (r0 starts as the input), so evolution should
// drive best fitness to ~1.0 almost immediately. If it doesn't, the loop
// is broken, not the problem.
static float identity(float x) { return x; }

int main() {
    std::cout << "=== LGP evolution run ===\n";
    std::cout << "POPULATION_SIZE   = " << LGPConfig::POPULATION_SIZE   << "\n";
    std::cout << "MAX_GENERATIONS   = " << LGPConfig::MAX_GENERATIONS   << "\n";
    std::cout << "ELITE_COUNT       = " << LGPConfig::ELITE_COUNT       << "\n";
    std::cout << "MAX_PROGRAM_SIZE  = " << LGPConfig::MAX_PROGRAM_SIZE  << "\n\n";

    // --- Sanity problem: f(x) = x -----------------------------------------
    {
        std::cout << "--- f(x) = x (sanity) ---\n";
        Dataset d = make_sr_dataset_1d(64, -2.0f, 2.0f, identity,42);
        LGPEngine engine;
        engine.evolve_sr(d);
        engine.print_history();
        std::cout << "\n";
        engine.print_best_program();
        std::cout << "\n";
    }

    // --- Real problem: f(x) = x*x + x -------------------------------------
    {
        std::cout << "--- f(x) = x*x + x (quadratic) ---\n";
        Dataset d = make_sr_dataset_1d(64, -2.0f, 2.0f, SRTargets::koza3, 42);
        LGPEngine engine;
        engine.evolve_sr(d);
        engine.print_history();
        std::cout << "\n";
        engine.print_best_program(); 
    }

    return 0;
}