#include "Evaluator.h"
#include "Interpreter.h"
#include "LGPConfig.h"
#include <algorithm> 
#include <cmath> 

namespace Evaluator{
    // new evaoluators will be added here
    // will need to carefully handle reduction here x ;-0 
    // Sr evaluator 
    float evaluate_sr_mse(const ProgramView& prog, const Dataset& dataset){
        constexpr int CHUNK = LGPConfig::NUM_CONTEXTS; // chunk is num contexts whichh for our case will be warp 
        const int N = dataset.N;
        const int num_inputs = dataset.num_inputs;
        // might consider chanign since we have smaller fitness cases 
        // Double accumulator: float SSE over 1000+ cases loses precision fast,
        // and an off-by-LSB MSE makes selection nondeterministic-looking when
        // two programs tie. Cheap to use double here; final cast is the cast.

        double sse = 0.0; // Sum squared error 
        float outputs[CHUNK]; // outpuct array containing all outputs of prog across dif contextts

        for (int base = 0; base < N; base+= CHUNK){
            // basically the interpreter gets a chunk every time, operates on that. but we only accumulate the valid output cols, the rest are just padded output 
            const int valid = std::min(CHUNK, N-base); // we either take the full chunk as valid, or the valid bits if chunk isnt full 

            Interpreter::run_stateless(
                prog.instructions,
                prog.length,
                &dataset.inputs[base  * num_inputs],  // current inputs for the each context - we mult the base (where we are at in the chunks) * the num inputs. To get the correct starting point for next input set for next chunk 
                num_inputs,
                outputs
            );// interpet one program on NUM contexts (Chunk)

            // sum cur error for chunk 
            for (int c = 0; c < valid ; ++c){
                //err - to get target we must add c to the current base... because it is based on the full sett of fitness cases not just one chhunk 
                const double err = double(outputs[c]) - double(dataset.targets[base + c]); // 
                sse += err * err; // accumulating the err^2
            }

        }
        // now we have gone through all chunks lets get the MSE and properly clamp
        return float(sse /double(N));


    }
}