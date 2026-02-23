#include "scheduler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

void LcmScheduler::init(int num_train_timesteps, float beta_start,
                         float beta_end) {
    num_train_timesteps_ = num_train_timesteps;

    // Scaled linear beta schedule (matching diffusers)
    // betas = linspace(sqrt(beta_start), sqrt(beta_end), N)^2
    std::vector<float> betas(num_train_timesteps);
    float sqrt_start = std::sqrt(beta_start);
    float sqrt_end = std::sqrt(beta_end);
    for (int i = 0; i < num_train_timesteps; i++) {
        float t = (float)i / (float)(num_train_timesteps - 1);
        float sqrt_beta = sqrt_start + t * (sqrt_end - sqrt_start);
        betas[i] = sqrt_beta * sqrt_beta;
    }

    // alphas_cumprod = cumprod(1 - betas)
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

    // Matches diffusers LCMScheduler.set_timesteps exactly:
    // 1. Create original timestep schedule
    int k = num_train_timesteps_ / original_inference_steps;
    std::vector<int> lcm_origin(original_inference_steps);
    for (int i = 0; i < original_inference_steps; i++) {
        lcm_origin[i] = (i + 1) * k - 1;
    }
    // = [19, 39, 59, ..., 999]

    // 2. Reverse
    std::reverse(lcm_origin.begin(), lcm_origin.end());
    // = [999, 979, 959, ..., 19]

    // 3. Select evenly spaced indices using linspace(0, N, num, endpoint=False)
    //    then floor to int — matches np.linspace + np.floor
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
    // Boundary condition scalings from LCM paper
    // sigma_data = 0.5, timestep_scaling = 10.0
    float scaled_t = (float)timestep * timestep_scaling_;
    float sd2 = sigma_data_ * sigma_data_;
    float st2 = scaled_t * scaled_t;

    c_skip = sd2 / (st2 + sd2);
    c_out = scaled_t / std::sqrt(st2 + sd2);
}

void LcmScheduler::step(const float* model_output, int timestep_idx,
                         float* sample, int num_elements, uint32_t seed) {
    int t = timesteps_[timestep_idx];

    // 1. Get alpha values
    float alpha_prod_t = alphas_cumprod_[t];

    float alpha_prod_t_prev;
    if (timestep_idx + 1 < num_inference_steps_) {
        int t_prev = timesteps_[timestep_idx + 1];
        alpha_prod_t_prev = alphas_cumprod_[t_prev];
    } else {
        // final_alpha_cumprod = 1.0 (set_alpha_to_one=True in config)
        alpha_prod_t_prev = 1.0f;
    }

    float sqrt_alpha_t = std::sqrt(alpha_prod_t);
    float sqrt_beta_t = std::sqrt(1.0f - alpha_prod_t);
    float sqrt_alpha_t_prev = std::sqrt(alpha_prod_t_prev);
    float sqrt_beta_t_prev = std::sqrt(1.0f - alpha_prod_t_prev);

    // 2. Get boundary condition scalings
    float c_skip, c_out;
    get_scalings(t, c_skip, c_out);

    // 3. Predict x0 (epsilon prediction type)
    // predicted_original_sample = (sample - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)

    // 4. Apply boundary conditions: denoised = c_out * x0_pred + c_skip * sample

    // 5. For non-final steps: prev = sqrt(alpha_prev) * denoised + sqrt(1-alpha_prev) * noise
    //    For final step: prev = denoised
    bool is_final = (timestep_idx == num_inference_steps_ - 1);

    // Generate noise for non-final steps
    std::vector<float> noise;
    if (!is_final) {
        noise.resize(num_elements);
        // Use deterministic noise based on seed + step index
        std::mt19937 gen(seed + timestep_idx + 1);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < num_elements; i++) {
            noise[i] = dist(gen);
        }
    }

    for (int i = 0; i < num_elements; i++) {
        // Predict x0
        float x0_pred = (sample[i] - sqrt_beta_t * model_output[i]) / sqrt_alpha_t;

        // Apply LCM boundary conditions
        float denoised = c_out * x0_pred + c_skip * sample[i];

        if (is_final) {
            sample[i] = denoised;
        } else {
            // Inject fresh noise (NOT model_output)
            sample[i] = sqrt_alpha_t_prev * denoised +
                         sqrt_beta_t_prev * noise[i];
        }
    }
}
