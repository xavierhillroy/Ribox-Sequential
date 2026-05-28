#include "Dataset.h"
#include <random>
#include "LGPConfig.h"
#include <cmath>
#include <fstream>

namespace SRTargets{
    // simple sr functions 
    float quadratic(float x ){return x*x + x;}
    float koza1(float x){return x*x*x*x + x*x*x +x*x +x;}
    float identity(float x) {return x;} 
    float koza3(float x) {
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x5 = x3 * x2;
        return x5 - 2.0f * x3 + x;
    }
}
namespace Nguyen{
    float Nguyen_1(float x) {return x*x*x + x*x + x;}
    float Nguyen_2(float x) {return x*x*x*x +x*x*x + x*x + x;}
    float Nguyen_3(float x) {return x*x*x*x*x + x*x*x*x +x*x*x + x*x + x;}
    float Nguyen_4(float x) {return x*x*x*x*x*x + x*x*x*x*x + x*x*x*x +x*x*x + x*x + x;}
    float Nguyen_5(float x) {return std::sin(x*x)*std::cos(x) -1;}
    float Nguyen_6(float x) {return std::sin(x) + sin(x+ x*x);}

    
}
// Compute SS_tot = Σ(y - ȳ)² over the N valid targets (excludes padding).
// Two-pass: mean first, then squared deviations. More numerically stable
// than the one-pass Σy² - (Σy)²/N form, which suffers catastrophic
// cancellation when the mean is large relative to the spread.
static double compute_ss_tot(const std::vector<float>& targets, int N) {
    double mean = 0.0;
    for (int i = 0; i < N; ++i) mean += double(targets[i]);
    mean /= double(N);

    double ss = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = double(targets[i]) - mean;
        ss += d * d;
    }
    return ss;
}

Dataset make_sr_dataset_1d(int N, float xmin, float xmax,float (*target_fn)(float), uint32_t seed){
    Dataset d;
    d.N = N; 
    d.num_inputs = 1; // since 1d

    // allocated padded inputs - padded to context size 32 ;)
    const int padded = d.padded_N();
    d.inputs.assign(padded * d.num_inputs, 0.0f);
    d.targets.assign(padded, 0.0f);

    //setting up RNG using global seed we had for repro
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(xmin,xmax);


    // filling up  vectors up to N (extra parts already padded to 0)
    for (int n = 0; n <N; ++n ){
        const float x = dist(rng);
        d.inputs[n] = x; // stride 1 because d.inputs is 1 
        d.targets[n] = target_fn(x);
    }
    d.ss_tot = compute_ss_tot(d.targets, N);  // cache once
    return d; // return finished dataset x 
}

void write_csv(const Dataset& d, const std::string& path) {
    std::ofstream f(path);
    f << "x,y\n";
    for (int n = 0; n < d.N; ++n)
        f << d.inputs[n] << "," << d.targets[n] << "\n";
}