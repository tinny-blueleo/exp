#pragma once

#include <cstdint>
#include <vector>

// LCM (Latent Consistency Model) noise scheduler
// Matches diffusers LCMScheduler exactly
class LcmScheduler {
  public:
    LcmScheduler() = default;

    // Configure the scheduler
    void init(int num_train_timesteps = 1000, float beta_start = 0.00085f,
              float beta_end = 0.012f);

    // Set inference timesteps for LCM
    void set_timesteps(int num_inference_steps,
                        int original_inference_steps = 50);

    // Single denoising step
    // model_output: predicted noise from UNet [N elements]
    // timestep_idx: index into timesteps_ (0-based)
    // sample: current noisy latents [N elements]
    // seed: RNG seed for noise injection (combined with timestep_idx)
    // Modifies sample in-place with denoised result
    void step(const float* model_output, int timestep_idx, float* sample,
              int num_elements, uint32_t seed);

    // Get timestep value for a given step index
    int timestep(int idx) const { return timesteps_[idx]; }

    // Number of inference steps
    int num_steps() const { return (int)timesteps_.size(); }

    const std::vector<int>& timesteps() const { return timesteps_; }

  private:
    // Boundary condition scalings for LCM
    void get_scalings(int timestep, float& c_skip, float& c_out) const;

    int num_train_timesteps_ = 1000;
    float timestep_scaling_ = 10.0f;
    float sigma_data_ = 0.5f;
    std::vector<float> alphas_cumprod_;
    std::vector<int> timesteps_;
    int num_inference_steps_ = 4;
};
