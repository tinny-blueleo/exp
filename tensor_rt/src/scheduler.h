#pragma once

#include <cstdint>
#include <vector>

// LCM (Latent Consistency Model) noise scheduler.
//
// In standard Stable Diffusion, denoising takes 20-50 small steps because
// each step can only remove a little noise. LCM is "distilled" — it was
// trained to take much bigger jumps, so it only needs 4 steps.
//
// The scheduler's job is to:
//   1. Choose which timesteps to use (e.g. [999, 759, 499, 259])
//   2. After each UNet prediction, combine the predicted clean image with
//      the right amount of noise for the next step
//
// The math is based on the "diffusion" formulation where:
//   - alphas_cumprod[t] tells you what fraction of the original image
//     remains at timestep t (1.0 = clean, ~0.0 = pure noise)
//   - The UNet predicts the noise component, which we subtract to
//     recover an estimate of the clean image (x0)
//   - LCM adds "boundary condition scalings" (c_skip, c_out) that
//     blend the UNet's prediction with the input to improve stability
//
// This implementation matches diffusers' LCMScheduler exactly.
class LcmScheduler {
  public:
    LcmScheduler() = default;

    // Precompute the alpha schedule. Called once at startup.
    //
    // num_train_timesteps: how many timesteps the model was trained with (1000)
    // beta_start/end: noise schedule endpoints — beta is how much noise is
    //   added at each step. SD uses a "scaled linear" schedule where beta
    //   increases quadratically from 0.00085 to 0.012.
    void init(int num_train_timesteps = 1000, float beta_start = 0.00085f,
              float beta_end = 0.012f);

    // Choose which timesteps to use for inference.
    //
    // num_inference_steps: how many denoising steps (4 for LCM)
    // original_inference_steps: the "full" schedule this was distilled from (50)
    //
    // For 4 steps with original=50, this produces timesteps [999, 759, 499, 259].
    // These are evenly spaced across the original 50-step schedule.
    void set_timesteps(int num_inference_steps,
                        int original_inference_steps = 50);

    // Perform one denoising step.
    //
    // model_output: noise predicted by the UNet for this timestep
    // timestep_idx: which step we're on (0, 1, 2, 3 for 4-step LCM)
    // sample: current noisy latents — MODIFIED IN PLACE with the result
    // num_elements: total floats in the latent tensor (1*4*64*64 = 16384)
    // seed: for generating fresh noise on non-final steps
    //
    // What happens each step:
    //   1. Predict x0 (clean image) = (sample - sqrt(1-α) * noise) / sqrt(α)
    //   2. Apply boundary conditions: denoised = c_out * x0 + c_skip * sample
    //   3. If not the last step, add back noise for the next timestep:
    //      next = sqrt(α_prev) * denoised + sqrt(1-α_prev) * fresh_noise
    //   4. If last step, just return denoised (no noise added)
    void step(const float* model_output, int timestep_idx, float* sample,
              int num_elements, uint32_t seed);

    int timestep(int idx) const { return timesteps_[idx]; }
    int num_steps() const { return (int)timesteps_.size(); }
    const std::vector<int>& timesteps() const { return timesteps_; }

  private:
    // Compute LCM boundary condition scalings for a given timestep.
    //
    // These come from the LCM paper and control how much of the UNet's
    // prediction vs the input sample to use. At high timesteps (noisy),
    // c_out is large (trust the prediction). At low timesteps (clean),
    // c_skip is large (trust the input).
    void get_scalings(int timestep, float& c_skip, float& c_out) const;

    int num_train_timesteps_ = 1000;

    // LCM-specific constants:
    // timestep_scaling: multiplier before computing boundary conditions (10.0)
    // sigma_data: assumed data standard deviation (0.5, from the LCM paper)
    float timestep_scaling_ = 10.0f;
    float sigma_data_ = 0.5f;

    // alphas_cumprod_[t] = cumulative product of (1 - beta) up to timestep t.
    // This represents the "signal strength" at each noise level:
    //   - alphas_cumprod_[0] ≈ 0.999 (almost clean)
    //   - alphas_cumprod_[999] ≈ 0.006 (almost pure noise)
    std::vector<float> alphas_cumprod_;

    // The specific timesteps to use during inference (e.g. [999, 759, 499, 259]).
    std::vector<int> timesteps_;
    int num_inference_steps_ = 4;
};
