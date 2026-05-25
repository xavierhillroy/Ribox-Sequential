#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <cstdint>
#include "LGPConfig.h"
// =============================================================================
// Dataset: inputs and targets for symbolic regression.
//
// Layout matches Interpreter::run_stateless:
//   inputs[n * num_inputs + i] = input i for fitness case n
//   targets[n]                  = ground truth for case n
//
// Stored row count is padded up to a multiple of NUM_CONTEXTS so the
// chunked evaluator can always issue a full 32-wide interpreter call
// without reading past the array end. Real case count is N. Padded rows
// are zero-filled and are never counted in the MSE sum -- the evaluator
// iterates only up to N and accumulates only the first `valid` outputs
// per chunk.
// =============================================================================

struct Dataset {
    std::vector<float> inputs; // Size = padded_N * num inputs (pads to context size (32 ;))- this is all to allign with warp level 
    std::vector<float> targets; /// size = padded_N
    int N; 
    int num_inputs; // num inputs of sr equation 

    int padded_N() const {
        const int c = LGPConfig::NUM_CONTEXTS;
        return((N + c -1)/c)*c ;
    }

};
namespace SRTargets {
    float quadratic(float x); // x^2 + x 
    float koza1(float x); // x^4 + x^3 + x^2 + x 
    float identity(float x); //f(x) = x;
    float koza3(float x);

}
// build a dataset by sampling unifromaly from [x_min, x_max] for a single inpout target fn- determinist function used from config 
Dataset make_sr_dataset_1d(int N, float xmin, float xmax, float (*target_fn)(float),uint32_t seed);
#endif
