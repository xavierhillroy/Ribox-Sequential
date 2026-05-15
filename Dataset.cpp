#include "Dataset.h"
#include <random>
#include "LGPConfig.h"

namespace SRTargets{
    // simple sr functions 
    float quadratic(floatt x ){return x*x + x;}
    float koza1(float x){return x*x*x*x + x*x*x +x*x +x;}
}
Dataset make_sr_dataset_1d(int N, float xmin, float xmax,float (*target_fn)(float),int num_inputs){
    Dataset d;
    d.N = N; 
    d.num_inputs = num_inputs; 

    // allocated padded inputs - padded to context size 32 ;)
    const int padded = d.padded_N();
    d.inputs.assign(padded * d.num_inputs, 0.0f);
    d.targets.assign(padded, 0.0f);

    //setting up RNG using global seed we had for repro
    std::mt19937 rng(LGPConfig::SEED);
    std::uniform_real_distribution<float> dist(xmin,xmax);


    // filling up  vectors up to N (extra parts already padded to 0)
    for (int n = 0; n <N; ++n ){
        const float x = dist(rng);
        d.inputs[n] = x; // stride 1 because d.inputs is 1 
        d.targets[n] = target_fn(x);
    }
    return d; // return finished dataset x 
}