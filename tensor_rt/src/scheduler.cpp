#include "scheduler.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

void LcmScheduler::init(int num_train_timesteps, float beta_start,
                         float beta_end) {
    num_train_timesteps_ = num_train_timesteps;

    // Linear beta schedule (scaled_linear in diffusers)
    // betas = linspace(sqrt(beta_start), sqrt(beta_end), num_train_timesteps)^2
    std::vector<float> betas(num_train_timesteps);
    float sqrt_start = std::sqrt(beta_start);
    float sqrt_end = std::sqrt(beta_end);
    for (int i = 0; i < num_train_timesteps; i++) {
        float t = (float)i / (float)(num_train_timesteps - 1);
        float sqrt_beta = sqrt_start + t * (sqrt_end - sqrt_start);
        betas[i] = sqrt_beta * sqrt_beta;
    }

    // alphas = 1 - betas
    // alphas_cumprod = cumprod(alphas)
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

    // LCM timestep schedule (matching diffusers LCMScheduler):
    // 1. Create original_inference_steps evenly-spaced timesteps
    int c = num_train_timesteps_ / original_inference_steps;
    std::vector<int> lcm_origin_timesteps(original_inference_steps);
    for (int i = 0; i < original_inference_steps; i++) {
        lcm_origin_timesteps[i] = (i + 1) * c - 1;
    }

    // 2. Subsample num_inference_steps from the original schedule
    int skipping_step = original_inference_steps / num_inference_steps;

    // Select indices: start from end, step back by skipping_step
    std::vector<int> selected_indices;
    for (int i = original_inference_steps - 1; i >= 0; i -= skipping_step) {
        selected_indices.push_back(i);
    }
    // Keep only num_inference_steps
    if ((int)selected_indices.size() > num_inference_steps) {
        selected_indices.resize(num_inference_steps);
    }

    // selected_indices is already in descending order (largest first)
    for (int idx : selected_indices) {
        timesteps_.push_back(lcm_origin_timesteps[idx]);
    }

    std::cout << "LCM timesteps (" << num_inference_steps << " steps): ";
    for (int t : timesteps_) std::cout << t << " ";
    std::cout << std::endl;
}

void LcmScheduler::step(const float* model_output, int timestep_idx,
                         float* sample, int num_elements) {
    int t = timesteps_[timestep_idx];

    // Get alpha values
    float alpha_prod_t = alphas_cumprod_[t];
    // For the previous timestep, use the next timestep in the list or final
    float alpha_prod_t_prev = 1.0f;
    if (timestep_idx + 1 < (int)timesteps_.size()) {
        int t_prev = timesteps_[timestep_idx + 1];
        alpha_prod_t_prev = alphas_cumprod_[t_prev];
    }

    float sqrt_alpha_t = std::sqrt(alpha_prod_t);
    float sqrt_one_minus_alpha_t = std::sqrt(1.0f - alpha_prod_t);
    float sqrt_alpha_t_prev = std::sqrt(alpha_prod_t_prev);
    float sqrt_one_minus_alpha_t_prev = std::sqrt(1.0f - alpha_prod_t_prev);

    // LCM step (DDIM-style, deterministic):
    // 1. Predict x0: x0 = (sample - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
    // 2. Compute prev: prev = sqrt(alpha_t_prev) * x0 + sqrt(1-alpha_t_prev) * noise_pred
    for (int i = 0; i < num_elements; i++) {
        float x0_pred =
            (sample[i] - sqrt_one_minus_alpha_t * model_output[i]) /
            sqrt_alpha_t;

        sample[i] = sqrt_alpha_t_prev * x0_pred +
                     sqrt_one_minus_alpha_t_prev * model_output[i];
    }
}
