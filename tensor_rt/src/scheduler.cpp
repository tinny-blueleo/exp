#include "scheduler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

void LcmScheduler::init(int num_train_timesteps, float beta_start,
                         float beta_end) {
    num_train_timesteps_ = num_train_timesteps;

    // Build the noise schedule: beta[t] controls how much noise is added at
    // each training timestep. Stable Diffusion uses a "scaled linear" schedule
    // where beta increases quadratically (sqrt then square) — this gives more
    // timesteps in the low-noise region where fine details matter.
    //
    //   betas = linspace(sqrt(beta_start), sqrt(beta_end), N) ^ 2
    //
    // beta_start=0.00085 → sqrt=0.0292, beta_end=0.012 → sqrt=0.1095
    std::vector<float> betas(num_train_timesteps);
    float sqrt_start = std::sqrt(beta_start);
    float sqrt_end = std::sqrt(beta_end);
    for (int i = 0; i < num_train_timesteps; i++) {
        float t = (float)i / (float)(num_train_timesteps - 1);
        float sqrt_beta = sqrt_start + t * (sqrt_end - sqrt_start);
        betas[i] = sqrt_beta * sqrt_beta;
    }

    // alphas_cumprod[t] = product of (1 - beta[0]) * (1 - beta[1]) * ... * (1 - beta[t])
    //
    // This tells us the "signal-to-noise ratio" at each timestep:
    //   - At t=0: alphas_cumprod ≈ 0.999 (image is almost unchanged)
    //   - At t=999: alphas_cumprod ≈ 0.006 (image is almost pure noise)
    //
    // During denoising, we use this to:
    //   - Extract the predicted clean image from the noisy input
    //   - Re-add the right amount of noise for the next step
    alphas_cumprod_.resize(num_train_timesteps);
    float prod = 1.0f;
    for (int i = 0; i < num_train_timesteps; i++) {
        prod *= (1.0f - betas[i]);
        alphas_cumprod_[i] = prod;
    }
}

void LcmScheduler::set_timesteps(int num_inference_steps,
                                   int original_inference_steps) {
    num_inference_steps_ = num_inference_steps;
    timesteps_.clear();

    // LCM timestep selection — matches diffusers LCMScheduler.set_timesteps.
    //
    // The idea: LCM was distilled from a model that used 50 steps. We pick
    // 4 evenly-spaced timesteps from that 50-step schedule.
    //
    // Step 1: Build the original 50-step schedule.
    //   k = 1000/50 = 20, so timesteps = [19, 39, 59, ..., 999]
    //   These are the timesteps the teacher model would have used.
    int k = num_train_timesteps_ / original_inference_steps;
    std::vector<int> lcm_origin(original_inference_steps);
    for (int i = 0; i < original_inference_steps; i++) {
        lcm_origin[i] = (i + 1) * k - 1;
    }

    // Step 2: Reverse to go from noisy → clean (denoising order).
    //   [999, 979, 959, ..., 19]
    std::reverse(lcm_origin.begin(), lcm_origin.end());

    // Step 3: Pick num_inference_steps evenly-spaced indices.
    //   For 4 steps from 50: indices 0, 12, 25, 37 → timesteps [999, 759, 499, 259]
    //   Uses np.linspace(0, N, num, endpoint=False) then floor.
    int n = original_inference_steps;
    for (int i = 0; i < num_inference_steps; i++) {
        float frac = (float)i * (float)n / (float)num_inference_steps;
        int idx = (int)frac;  // floor
        timesteps_.push_back(lcm_origin[idx]);
    }

    std::cout << "LCM timesteps (" << num_inference_steps << " steps): ";
    for (int t : timesteps_) std::cout << t << " ";
    std::cout << std::endl;
}

void LcmScheduler::get_scalings(int timestep, float& c_skip,
                                  float& c_out) const {
    // Boundary condition scalings from the LCM/Consistency Model paper.
    //
    // These control how much we trust the model's prediction vs the input:
    //   c_skip = σ_data² / (scaled_t² + σ_data²)  — weight on input sample
    //   c_out  = scaled_t / sqrt(scaled_t² + σ_data²) — weight on prediction
    //
    // At high timesteps (noisy): c_out ≈ 1, c_skip ≈ 0 (trust prediction)
    // At low timesteps (clean):  c_out ≈ 0, c_skip ≈ 1 (trust input)
    //
    // σ_data = 0.5, timestep_scaling = 10.0 (from LCM config)
    float scaled_t = (float)timestep * timestep_scaling_;
    float sd2 = sigma_data_ * sigma_data_;
    float st2 = scaled_t * scaled_t;

    c_skip = sd2 / (st2 + sd2);
    c_out = scaled_t / std::sqrt(st2 + sd2);
}

void LcmScheduler::step(const float* model_output, int timestep_idx,
                         float* sample, int num_elements, uint32_t seed) {
    int t = timesteps_[timestep_idx];

    // --- Get alpha (signal strength) for current and previous timesteps ---
    //
    // alpha_prod_t: signal strength at current timestep
    // alpha_prod_t_prev: signal strength at the NEXT timestep we'll jump to
    //   (confusingly called "prev" because diffusion goes backwards in time)
    //
    // For the final step, alpha_prod_t_prev = 1.0 (fully clean, no noise).
    float alpha_prod_t = alphas_cumprod_[t];

    float alpha_prod_t_prev;
    if (timestep_idx + 1 < num_inference_steps_) {
        int t_prev = timesteps_[timestep_idx + 1];
        alpha_prod_t_prev = alphas_cumprod_[t_prev];
    } else {
        // Final step targets a completely clean image (set_alpha_to_one=True).
        alpha_prod_t_prev = 1.0f;
    }

    // Precompute square roots used in the denoising formula:
    //   sqrt(α) = signal amplitude, sqrt(1-α) = noise amplitude
    float sqrt_alpha_t = std::sqrt(alpha_prod_t);
    float sqrt_beta_t = std::sqrt(1.0f - alpha_prod_t);
    float sqrt_alpha_t_prev = std::sqrt(alpha_prod_t_prev);
    float sqrt_beta_t_prev = std::sqrt(1.0f - alpha_prod_t_prev);

    // Boundary condition scalings (see get_scalings comment above).
    float c_skip, c_out;
    get_scalings(t, c_skip, c_out);

    // For non-final steps, we add fresh random noise to the denoised result.
    // This is critical: we use NEW noise, not the model's predicted noise.
    // The seed is deterministic (base_seed + step_index) for reproducibility.
    bool is_final = (timestep_idx == num_inference_steps_ - 1);

    std::vector<float> noise;
    if (!is_final) {
        noise.resize(num_elements);
        std::mt19937 gen(seed + timestep_idx + 1);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < num_elements; i++) {
            noise[i] = dist(gen);
        }
    }

    // --- The core denoising loop (per-element) ---
    for (int i = 0; i < num_elements; i++) {
        // 1. Predict the clean image (x0) from the noisy sample.
        //    The UNet predicted the noise (ε), so:
        //      x0 = (noisy_sample - sqrt(1-α) * ε) / sqrt(α)
        //    This is just rearranging: noisy = sqrt(α)*clean + sqrt(1-α)*noise
        float x0_pred = (sample[i] - sqrt_beta_t * model_output[i]) / sqrt_alpha_t;

        // 2. Apply LCM boundary conditions to stabilize the prediction.
        //      denoised = c_out * x0_pred + c_skip * sample
        //    This blends the raw prediction with the input, preventing the
        //    model from making wild jumps early in denoising.
        float denoised = c_out * x0_pred + c_skip * sample[i];

        if (is_final) {
            // Last step: just output the denoised result, no more noise.
            sample[i] = denoised;
        } else {
            // 3. Re-noise for the next step: jump to the noise level of the
            //    next timestep by mixing denoised image with fresh noise.
            //      next = sqrt(α_prev) * denoised + sqrt(1-α_prev) * noise
            sample[i] = sqrt_alpha_t_prev * denoised +
                         sqrt_beta_t_prev * noise[i];
        }
    }
}
