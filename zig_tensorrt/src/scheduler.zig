// LCM (Latent Consistency Model) noise scheduler — pure Zig, no dependencies.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  HOW DIFFUSION DENOISING WORKS                                         │
// │                                                                         │
// │  Imagine a photograph being gradually corrupted by static noise:        │
// │                                                                         │
// │    clean image ──(add noise)──> slightly noisy ──> ... ──> pure static  │
// │       t=0                           t=100              t=999            │
// │                                                                         │
// │  This is the "forward process" — trivial, just add Gaussian noise.      │
// │  The timestep t tells you how much noise was added.                     │
// │                                                                         │
// │  The neural network (UNet) learns the REVERSE: given a noisy image and  │
// │  a timestep, predict what noise was added. Then we subtract it.         │
// │                                                                         │
// │    pure static ──(UNet predicts noise, subtract)──> ... ──> clean image │
// │       t=999                                                 t=0        │
// │                                                                         │
// │  Standard Stable Diffusion takes 20-50 small denoising steps.           │
// │  LCM (Latent Consistency Model) is "distilled" — trained to take much   │
// │  bigger jumps — so it only needs 4 steps.                               │
// │                                                                         │
// │  The scheduler's job:                                                   │
// │    1. Choose which timesteps to use (e.g. [999, 759, 499, 259])         │
// │    2. After each UNet prediction, compute a cleaner version of the      │
// │       latents using the predicted noise and add back fresh noise for    │
// │       the next step.                                                    │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ABOUT THE MODEL: LCM Dreamshaper v7
//
//   This scheduler is designed for LCM Dreamshaper v7, which has this lineage:
//
//   Stable Diffusion 1.5 (Stability AI, base model)
//       └── Dreamshaper v7 (community fine-tune, full checkpoint — NOT a LoRA)
//           └── LCM Dreamshaper v7 (consistency distillation by SimianLuo)
//
//   Dreamshaper is a full SD 1.5 fine-tune trained on curated artistic data
//   for better aesthetics. It's ~3.3GB — the same size as the base model.
//   LoRA adapters are different: they're small (~67MB) add-on files that modify
//   a base model's behavior without replacing it. LCM-LoRA variants exist
//   separately, but this model is a full standalone distilled checkpoint.
//
//   The LCM distillation process trained the model to produce good images in
//   4 steps instead of 20-50, using a "consistency" training objective that
//   ensures different noise levels map to the same clean image.

const std = @import("std");

const max_train_timesteps = 1000;
const max_inference_steps = 50;

pub const LcmScheduler = struct {
    num_train_timesteps: i32 = 1000,
    timestep_scaling: f32 = 10.0,
    sigma_data: f32 = 0.5,

    // alphas_cumprod[t] = cumulative product of (1 - beta) up to timestep t.
    // Represents "signal strength" at each noise level:
    //   alphas_cumprod[0] ≈ 0.999 (almost clean)
    //   alphas_cumprod[999] ≈ 0.006 (almost pure noise)
    alphas_cumprod: [max_train_timesteps]f32 = undefined,

    // The specific timesteps for inference (e.g. [999, 759, 499, 259]).
    timesteps_buf: [max_inference_steps]i32 = undefined,
    timesteps_len: usize = 0,
    num_inference_steps: i32 = 4,

    /// Precompute the alpha schedule. Called once at startup.
    ///
    /// The noise schedule defines how much noise is added at each training
    /// timestep. SD uses a "scaled linear" schedule where beta (noise amount)
    /// increases quadratically — giving more timesteps in the low-noise region
    /// where fine details matter.
    pub fn init(self: *LcmScheduler) void {
        self.initWithParams(1000, 0.00085, 0.012);
    }

    pub fn initWithParams(self: *LcmScheduler, num_train: i32, beta_start: f32, beta_end: f32) void {
        self.num_train_timesteps = num_train;
        const n: usize = @intCast(num_train);

        // betas = linspace(sqrt(beta_start), sqrt(beta_end), N) ^ 2
        const sqrt_start = @sqrt(beta_start);
        const sqrt_end = @sqrt(beta_end);

        // alphas_cumprod[t] = (1-beta[0]) * (1-beta[1]) * ... * (1-beta[t])
        var prod: f32 = 1.0;
        for (0..n) |i| {
            const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1));
            const sqrt_beta = sqrt_start + t * (sqrt_end - sqrt_start);
            const beta = sqrt_beta * sqrt_beta;
            prod *= (1.0 - beta);
            self.alphas_cumprod[i] = prod;
        }
    }

    /// Choose which timesteps to use for inference.
    ///
    /// LCM was distilled from a 50-step schedule. We pick num_steps evenly
    /// spaced timesteps from that original schedule.
    /// For 4 steps: [999, 759, 499, 259]
    pub fn setTimesteps(self: *LcmScheduler, num_steps: i32) void {
        self.setTimestepsEx(num_steps, 50);
    }

    pub fn setTimestepsEx(self: *LcmScheduler, num_steps: i32, original_steps: i32) void {
        self.num_inference_steps = num_steps;
        self.timesteps_len = 0;

        // Build the original schedule: [19, 39, 59, ..., 999] then reverse
        const k = @divTrunc(self.num_train_timesteps, original_steps);
        const n: usize = @intCast(original_steps);

        var lcm_origin: [max_inference_steps]i32 = undefined;
        for (0..n) |i| {
            lcm_origin[i] = @as(i32, @intCast(i + 1)) * k - 1;
        }
        // Reverse to denoising order: [999, 979, ..., 19]
        std.mem.reverse(i32, lcm_origin[0..n]);

        // Pick evenly spaced indices (matches np.linspace endpoint=False + floor)
        const steps: usize = @intCast(num_steps);
        for (0..steps) |i| {
            const frac = @as(f32, @floatFromInt(i)) * @as(f32, @floatFromInt(n)) / @as(f32, @floatFromInt(steps));
            const idx: usize = @intFromFloat(frac);
            self.timesteps_buf[self.timesteps_len] = lcm_origin[idx];
            self.timesteps_len += 1;
        }

        // Log timesteps
        std.debug.print("LCM timesteps ({d} steps):", .{num_steps});
        for (self.timesteps_buf[0..self.timesteps_len]) |t| {
            std.debug.print(" {d}", .{t});
        }
        std.debug.print("\n", .{});
    }

    pub fn timestep(self: *const LcmScheduler, idx: usize) i32 {
        return self.timesteps_buf[idx];
    }

    pub fn numSteps(self: *const LcmScheduler) usize {
        return self.timesteps_len;
    }

    /// Perform one denoising step.
    ///
    /// Given the UNet's noise prediction and the current noisy latents,
    /// compute cleaner latents:
    ///   1. Predict the clean image (x0) by removing predicted noise
    ///   2. Apply LCM boundary conditions for stability
    ///   3. If not the last step, add fresh noise for the next timestep
    pub fn step(self: *const LcmScheduler, model_output: []const f32, timestep_idx: usize, sample: []f32, seed: u32) void {
        const t: usize = @intCast(self.timesteps_buf[timestep_idx]);

        // Alpha values: signal strength at current and next timestep.
        // "prev" is confusing — it means the NEXT timestep in denoising order
        // (lower noise level). For the final step, we target alpha=1.0 (clean).
        const alpha_prod_t = self.alphas_cumprod[t];
        const alpha_prod_t_prev: f32 = if (timestep_idx + 1 < self.timesteps_len)
            self.alphas_cumprod[@intCast(self.timesteps_buf[timestep_idx + 1])]
        else
            1.0; // Final step targets fully clean

        // sqrt(alpha) = signal amplitude, sqrt(1-alpha) = noise amplitude
        const sqrt_alpha_t = @sqrt(alpha_prod_t);
        const sqrt_beta_t = @sqrt(1.0 - alpha_prod_t);
        const sqrt_alpha_t_prev = @sqrt(alpha_prod_t_prev);
        const sqrt_beta_t_prev = @sqrt(1.0 - alpha_prod_t_prev);

        // LCM boundary condition scalings (from consistency model paper).
        // c_skip blends the input, c_out blends the prediction.
        const scalings = getScalings(self, @intCast(t));

        const is_final = (timestep_idx == self.timesteps_len - 1);

        // Generate fresh random noise for non-final steps.
        // IMPORTANT: we use NEW noise, not the model's predicted noise.
        // Seeded deterministically for reproducibility.
        var noise: [1 * 4 * 64 * 64]f32 = undefined;
        if (!is_final) {
            // Seed with base_seed + step_index + 1 to get different noise each step
            var prng = std.Random.DefaultPrng.init(@as(u64, seed) + @as(u64, @intCast(timestep_idx)) + 1);
            const rand = prng.random();
            for (0..sample.len) |i| {
                // Box-Muller transform: convert uniform [0,1) to standard normal
                const r1 = rand.float(f32);
                const r2 = rand.float(f32);
                noise[i] = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
            }
        }

        // Core denoising loop (per element):
        for (0..sample.len) |i| {
            // 1. Predict clean image: x0 = (noisy - sqrt(1-α)*noise) / sqrt(α)
            //    This rearranges: noisy = sqrt(α)*clean + sqrt(1-α)*noise
            const x0_pred = (sample[i] - sqrt_beta_t * model_output[i]) / sqrt_alpha_t;

            // 2. Apply LCM boundary conditions for stability
            const denoised = scalings.c_out * x0_pred + scalings.c_skip * sample[i];

            if (is_final) {
                // Last step: output clean result
                sample[i] = denoised;
            } else {
                // 3. Re-noise for next step: mix clean estimate with fresh noise
                sample[i] = sqrt_alpha_t_prev * denoised + sqrt_beta_t_prev * noise[i];
            }
        }
    }

    const Scalings = struct { c_skip: f32, c_out: f32 };

    /// Boundary condition scalings from the LCM/Consistency Model paper.
    /// At high timesteps (noisy): c_out ≈ 1, c_skip ≈ 0 (trust prediction)
    /// At low timesteps (clean): c_out ≈ 0, c_skip ≈ 1 (trust input)
    fn getScalings(self: *const LcmScheduler, t: i32) Scalings {
        const scaled_t: f32 = @as(f32, @floatFromInt(t)) * self.timestep_scaling;
        const sd2 = self.sigma_data * self.sigma_data;
        const st2 = scaled_t * scaled_t;
        return .{
            .c_skip = sd2 / (st2 + sd2),
            .c_out = scaled_t / @sqrt(st2 + sd2),
        };
    }
};
