// Stable Diffusion inference pipeline — orchestrates the three neural networks.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │  THE FULL DIFFUSION PIPELINE                                            │
// │                                                                         │
// │  Text-to-image generation works in three phases:                        │
// │                                                                         │
// │  Phase 1: CLIP Text Encoding                                            │
// │    "Taj Mahal in space" → [1, 77, 768] float tensor                    │
// │    Converts human-readable text into a dense numerical representation   │
// │    that the UNet understands. CLIP was trained on 400M image-text pairs │
// │    by OpenAI, learning a shared "meaning space" between text and images.│
// │    Each of the 77 token positions gets a 768-dim vector encoding its    │
// │    contextual meaning (via transformer self-attention).                 │
// │                                                                         │
// │  Phase 2: UNet Denoising (the actual image generation)                  │
// │    Random noise [1,4,64,64] → 4 UNet steps → clean latents [1,4,64,64]│
// │    Starting from pure Gaussian noise, the UNet repeatedly predicts and  │
// │    removes noise, guided by the text embeddings. Each step, the UNet    │
// │    uses cross-attention to match spatial features with text concepts    │
// │    ("Taj Mahal" → building shapes, "space" → starfield patterns).      │
// │                                                                         │
// │    This operates in "latent space" — a compressed 64x64x4 encoding     │
// │    (16K values) rather than full 512x512x3 pixels (786K values).       │
// │    Working in this compact space is what makes diffusion tractable.     │
// │                                                                         │
// │  Phase 3: VAE Decode                                                    │
// │    Clean latents [1,4,64,64] → RGB image [1,3,512,512]                │
// │    The Variational Autoencoder decoder expands the compressed latent    │
// │    representation back into a full-resolution image. It's a learned    │
// │    neural upsampler that understands image structure.                   │
// │                                                                         │
// │  ABOUT LCM DREAMSHAPER V7                                               │
// │                                                                         │
// │  The model we use is LCM Dreamshaper v7. Its lineage:                   │
// │                                                                         │
// │    Stable Diffusion 1.5 (Stability AI — the base model)                │
// │       └── Dreamshaper v7 (community fine-tune of SD 1.5)               │
// │           └── LCM Dreamshaper v7 (consistency distillation)            │
// │                                                                         │
// │  Dreamshaper v7 is a FULL CHECKPOINT fine-tune (~3.3GB), NOT a LoRA.   │
// │  It was trained on curated artistic data for better aesthetics.         │
// │  LoRA adapters are small (~67MB) add-on files that modify a base model │
// │  without replacing it — a different technique. LCM-LoRA variants exist │
// │  separately, but this model is a full standalone distilled checkpoint.  │
// │                                                                         │
// │  LCM (Latent Consistency Model) distillation trained it to produce     │
// │  good images in just 4 steps instead of the usual 20-50, by learning   │
// │  to take much bigger denoising jumps. It also bakes in "classifier-    │
// │  free guidance" via a guidance scale embedding, eliminating the need    │
// │  for the expensive two-pass trick that doubles inference cost.          │
// └─────────────────────────────────────────────────────────────────────────┘

const std = @import("std");
const eng = @import("../engine.zig");
const LcmScheduler = @import("scheduler.zig").LcmScheduler;
const ClipTokenizer = @import("tokenizer.zig").ClipTokenizer;

const Engine = eng.Engine;
const DataType = eng.DataType;
const TensorInfo = eng.TensorInfo;

// ── FP16 ↔ FP32 conversion ─────────────────────────────────────────────
// TensorRT engines built with FP16 may have half-precision I/O tensors.
// We work in FP32 on the CPU side and convert at the GPU boundary.

fn fp32ToFp16(value: f32) u16 {
    const f: u32 = @bitCast(value);
    const sign: u32 = (f >> 16) & 0x8000;
    const exp_raw: i32 = @as(i32, @intCast((f >> 23) & 0xFF)) - 127 + 15;
    const frac: u32 = (f >> 13) & 0x3FF;
    if (exp_raw <= 0) return @intCast(sign);
    if (exp_raw >= 31) return @intCast(sign | 0x7C00);
    return @intCast(sign | (@as(u32, @intCast(exp_raw)) << 10) | frac);
}

fn fp16ToFp32(h: u16) f32 {
    const sign: u32 = (@as(u32, h) & 0x8000) << 16;
    var exp: u32 = (@as(u32, h) >> 10) & 0x1F;
    var frac: u32 = @as(u32, h) & 0x3FF;
    if (exp == 0) {
        if (frac == 0) return @bitCast(sign);
        exp = 1;
        while (frac & 0x400 == 0) {
            frac <<= 1;
            exp -%= 1;
        }
        frac &= 0x3FF;
    } else if (exp == 31) {
        return @bitCast(sign | 0x7F800000 | (frac << 13));
    }
    return @bitCast(sign | ((exp + 127 - 15) << 23) | (frac << 13));
}

// ── Guidance scale embedding ────────────────────────────────────────────
// LCM bakes classifier-free guidance into the model. Instead of running
// the UNet twice (conditioned + unconditioned) and blending, LCM takes
// a guidance scale value (e.g. 8.0) and encodes it as a 256-dim sinusoidal
// vector — the same kind of positional encoding used for timesteps.
// Computed once and reused for every denoising step.

fn getGuidanceScaleEmbedding(w: f32) [256]f32 {
    const scaled_w = w * 1000.0;
    const half = 128;
    const log_base = @log(@as(f32, 10000.0)) / @as(f32, @floatFromInt(half - 1));

    var emb: [256]f32 = undefined;
    for (0..half) |i| {
        const freq = @exp(@as(f32, @floatFromInt(i)) * -log_base);
        emb[i] = @sin(scaled_w * freq);
        emb[half + i] = @cos(scaled_w * freq);
    }
    return emb;
}

fn nowMs() f64 {
    const ts = std.time.Instant.now() catch return 0;
    return @as(f64, @floatFromInt(ts.timestamp.sec)) * 1000.0 +
        @as(f64, @floatFromInt(ts.timestamp.nsec)) / 1_000_000.0;
}

pub const Pipeline = struct {
    allocator: std.mem.Allocator,

    // Borrowed pointers — Pipeline does NOT own these.
    // Ownership belongs to InferenceModels (see inference.zig).
    text_encoder: *Engine,
    unet: *Engine,
    vae_decoder: *Engine,
    tokenizer: *ClipTokenizer,
    scheduler: *LcmScheduler,

    // Engine paths needed for sequential mode reload
    text_encoder_path: [*:0]const u8,
    unet_path: [*:0]const u8,
    vae_decoder_path: [*:0]const u8,

    sequential_mode: bool = false,

    pub fn init(
        allocator: std.mem.Allocator,
        text_encoder: *Engine,
        unet: *Engine,
        vae_decoder: *Engine,
        tokenizer: *ClipTokenizer,
        scheduler: *LcmScheduler,
        text_encoder_path: [*:0]const u8,
        unet_path: [*:0]const u8,
        vae_decoder_path: [*:0]const u8,
        sequential_mode: bool,
    ) Pipeline {
        return .{
            .allocator = allocator,
            .text_encoder = text_encoder,
            .unet = unet,
            .vae_decoder = vae_decoder,
            .tokenizer = tokenizer,
            .scheduler = scheduler,
            .text_encoder_path = text_encoder_path,
            .unet_path = unet_path,
            .vae_decoder_path = vae_decoder_path,
            .sequential_mode = sequential_mode,
        };
    }

    pub fn generate(self: *Pipeline, prompt: []const u8, seed: u32, num_steps: i32) ![]u8 {
        const t0 = nowMs();

        std.debug.print("\nGenerating: \"{s}\" (seed={d}, steps={d})\n", .{ prompt, seed, num_steps });

        const embeddings = try self.encodeText(prompt);
        defer self.allocator.free(embeddings);

        const latents = try self.denoise(embeddings, seed, num_steps);
        defer self.allocator.free(latents);

        const pixels = try self.decodeLatents(latents);

        std.debug.print("\nTotal generation: {d:.1} ms\n", .{nowMs() - t0});
        return pixels;
    }

    // ── Phase 1: Text Encoding ─────────────────────────────────────────
    // The CLIP text encoder converts a text prompt into dense embeddings.
    // These 77 vectors of 768 floats encode the contextual meaning of
    // each token, learned from 400 million image-text pairs.

    fn encodeText(self: *Pipeline, prompt: []const u8) ![]f32 {
        const t0 = nowMs();


        if (self.sequential_mode and !self.text_encoder.isLoaded())
            try self.text_encoder.load(self.text_encoder_path);

        // Tokenize: text → 77 integer token IDs
        var token_ids: [77]i32 = undefined;
        self.tokenizer.encode(prompt, &token_ids);

        std.debug.print("Tokens:", .{});
        for (0..@min(10, 77)) |i| {
            std.debug.print(" {d}", .{token_ids[i]});
        }
        std.debug.print(" ... (77 total)\n", .{});

        // Set shape and upload tokens to GPU
        try self.text_encoder.setInputShape("input_ids", &[_]i64{ 1, 77 });
        try self.text_encoder.setInput("input_ids", std.mem.sliceAsBytes(&token_ids));

        try self.text_encoder.infer();

        // Read back embeddings [1, 77, 768]
        const embed_size: usize = 1 * 77 * 768;
        const embeddings = try self.allocator.alloc(f32, embed_size);

        const out = self.text_encoder.findOutput("last_hidden_state") orelse
            try self.text_encoder.getOutputInfo(0);

        if (out.dtype == .float16) {
            const fp16_buf = try self.allocator.alloc(u16, embed_size);
            defer self.allocator.free(fp16_buf);
            try self.text_encoder.getOutput(out.name, std.mem.sliceAsBytes(fp16_buf));
            for (0..embed_size) |i| {
                embeddings[i] = fp16ToFp32(fp16_buf[i]);
            }
        } else {
            try self.text_encoder.getOutput(out.name, std.mem.sliceAsBytes(embeddings));
        }

        if (self.sequential_mode) self.text_encoder.unload();

        std.debug.print("Text encoding: {d:.1} ms\n", .{nowMs() - t0});
        return embeddings;
    }

    // ── Phase 2: Denoising (UNet) ──────────────────────────────────────
    // Start with pure Gaussian noise and iteratively denoise, guided by
    // the text embeddings. Each step:
    //   1. UNet receives noisy latents + timestep + text embeddings
    //   2. It predicts the noise component
    //   3. Scheduler subtracts noise and optionally adds fresh noise

    fn denoise(self: *Pipeline, text_embeddings: []const f32, seed: u32, num_steps: i32) ![]f32 {
        const t0 = nowMs();


        if (self.sequential_mode and !self.unet.isLoaded())
            try self.unet.load(self.unet_path);

        self.scheduler.setTimesteps(num_steps);

        // Initialize latents with random noise from seed
        const latent_size: usize = 1 * 4 * 64 * 64;
        const latents = try self.allocator.alloc(f32, latent_size);
        {
            var prng = std.Random.DefaultPrng.init(seed);
            const rand = prng.random();
            for (0..latent_size) |i| {
                const r1 = rand.float(f32);
                const r2 = rand.float(f32);
                latents[i] = @sqrt(-2.0 * @log(r1)) * @cos(2.0 * std.math.pi * r2);
            }
        }

        // Guidance scale = 8.0: how strongly to follow the text prompt.
        // Higher = more literal, lower = more creative.
        const w_embedding = getGuidanceScaleEmbedding(8.0);

        // Detect engine I/O properties
        const has_timestep_cond = self.unet.findInput("timestep_cond") != null;
        const sample_info = self.unet.findInput("sample");
        const unet_fp16 = if (sample_info) |s| s.dtype == .float16 else false;

        // Set concrete shapes for dynamic-shape engine
        try self.unet.setInputShape("sample", &[_]i64{ 1, 4, 64, 64 });
        try self.unet.setInputShape("timestep", &[_]i64{1});
        try self.unet.setInputShape("encoder_hidden_states", &[_]i64{ 1, 77, 768 });
        if (has_timestep_cond)
            try self.unet.setInputShape("timestep_cond", &[_]i64{ 1, 256 });

        // Upload constant inputs (same for every step)
        if (has_timestep_cond) {
            const cond_info = self.unet.findInput("timestep_cond").?;
            if (cond_info.dtype == .float16) {
                var fp16: [256]u16 = undefined;
                for (0..256) |i| fp16[i] = fp32ToFp16(w_embedding[i]);
                try self.unet.setInput("timestep_cond", std.mem.sliceAsBytes(&fp16));
            } else {
                try self.unet.setInput("timestep_cond", std.mem.sliceAsBytes(&w_embedding));
            }
        }

        // Upload text embeddings
        const embed_size = text_embeddings.len;
        if (unet_fp16) {
            const fp16_buf = try self.allocator.alloc(u16, embed_size);
            defer self.allocator.free(fp16_buf);
            for (0..embed_size) |i| fp16_buf[i] = fp32ToFp16(text_embeddings[i]);
            try self.unet.setInput("encoder_hidden_states", std.mem.sliceAsBytes(fp16_buf));
        } else {
            try self.unet.setInput("encoder_hidden_states", std.mem.sliceAsBytes(text_embeddings));
        }

        const unet_out_info = self.unet.findOutput("out_sample") orelse
            try self.unet.getOutputInfo(0);

        // ── Denoising loop ──
        const noise_pred = try self.allocator.alloc(f32, latent_size);
        defer self.allocator.free(noise_pred);

        for (0..self.scheduler.numSteps()) |i| {
            const step_t0 = nowMs();
            const t = self.scheduler.timestep(i);

            // Upload current noisy latents
            if (unet_fp16) {
                const fp16_buf = try self.allocator.alloc(u16, latent_size);
                defer self.allocator.free(fp16_buf);
                for (0..latent_size) |j| fp16_buf[j] = fp32ToFp16(latents[j]);
                try self.unet.setInput("sample", std.mem.sliceAsBytes(fp16_buf));
            } else {
                try self.unet.setInput("sample", std.mem.sliceAsBytes(latents));
            }

            // Upload timestep (tells UNet the current noise level)
            var timestep_val: i64 = t;
            try self.unet.setInput("timestep", std.mem.asBytes(&timestep_val));

            try self.unet.infer();

            // Read predicted noise back from GPU
            if (unet_out_info.dtype == .float16) {
                const fp16_buf = try self.allocator.alloc(u16, latent_size);
                defer self.allocator.free(fp16_buf);
                try self.unet.getOutput(unet_out_info.name, std.mem.sliceAsBytes(fp16_buf));
                for (0..latent_size) |j| noise_pred[j] = fp16ToFp32(fp16_buf[j]);
            } else {
                try self.unet.getOutput(unet_out_info.name, std.mem.sliceAsBytes(noise_pred));
            }

            // Scheduler: subtract predicted noise, apply boundary conditions,
            // inject fresh noise for non-final steps
            self.scheduler.step(noise_pred, i, latents, seed);

            std.debug.print("  Step {d}/{d} (t={d}): {d:.1} ms\n", .{
                i + 1, @as(usize, @intCast(num_steps)), t, nowMs() - step_t0,
            });
        }

        if (self.sequential_mode) self.unet.unload();

        std.debug.print("Denoising: {d:.1} ms\n", .{nowMs() - t0});
        return latents;
    }

    // ── Phase 3: VAE Decode ────────────────────────────────────────────
    // The VAE decoder expands 64x64x4 latents into 512x512x3 RGB pixels.
    //
    // Scaling factor: during training, the VAE encoder multiplied latent
    // values by 0.18215. We must undo this before decoding. The HuggingFace
    // ONNX model expects pre-scaled input (unlike some custom exports that
    // bake the scaling into the graph).

    fn decodeLatents(self: *Pipeline, latents: []f32) ![]u8 {
        const t0 = nowMs();


        if (self.sequential_mode and !self.vae_decoder.isLoaded())
            try self.vae_decoder.load(self.vae_decoder_path);

        const latent_size = latents.len;

        // Undo VAE encoder scaling factor
        const vae_scaling_factor: f32 = 0.18215;
        const scaled = try self.allocator.alloc(f32, latent_size);
        defer self.allocator.free(scaled);
        for (0..latent_size) |i| {
            scaled[i] = latents[i] / vae_scaling_factor;
        }

        const inp = self.vae_decoder.findInput("latent_sample") orelse
            try self.vae_decoder.getInputInfo(0);

        try self.vae_decoder.setInputShape(inp.name, &[_]i64{ 1, 4, 64, 64 });

        if (inp.dtype == .float16) {
            const fp16_buf = try self.allocator.alloc(u16, latent_size);
            defer self.allocator.free(fp16_buf);
            for (0..latent_size) |i| fp16_buf[i] = fp32ToFp16(scaled[i]);
            try self.vae_decoder.setInput(inp.name, std.mem.sliceAsBytes(fp16_buf));
        } else {
            try self.vae_decoder.setInput(inp.name, std.mem.sliceAsBytes(scaled));
        }

        try self.vae_decoder.infer();

        // Read decoded image: [1, 3, 512, 512] in CHW format, range [-1, 1]
        const img_elements: usize = 1 * 3 * 512 * 512;
        const raw = try self.allocator.alloc(f32, img_elements);
        defer self.allocator.free(raw);

        const out = self.vae_decoder.findOutput("sample") orelse
            try self.vae_decoder.getOutputInfo(0);

        if (out.dtype == .float16) {
            const fp16_buf = try self.allocator.alloc(u16, img_elements);
            defer self.allocator.free(fp16_buf);
            try self.vae_decoder.getOutput(out.name, std.mem.sliceAsBytes(fp16_buf));
            for (0..img_elements) |i| raw[i] = fp16ToFp32(fp16_buf[i]);
        } else {
            try self.vae_decoder.getOutput(out.name, std.mem.sliceAsBytes(raw));
        }

        if (self.sequential_mode) self.vae_decoder.unload();

        // Convert CHW float [-1,1] → HWC uint8 [0,255]
        // CHW = channels-first: all red values, then green, then blue
        // HWC = interleaved: RGBRGBRGB... (what PNG expects)
        const H: usize = 512;
        const W: usize = 512;
        const C: usize = 3;
        const pixels = try self.allocator.alloc(u8, H * W * C);

        for (0..H) |y| {
            for (0..W) |x| {
                for (0..C) |ch| {
                    var val = raw[ch * H * W + y * W + x]; // CHW index
                    val = (val + 1.0) * 0.5 * 255.0; // [-1,1] → [0,255]
                    val = std.math.clamp(val, 0.0, 255.0);
                    pixels[(y * W + x) * C + ch] = @intFromFloat(val + 0.5);
                }
            }
        }

        std.debug.print("VAE decode: {d:.1} ms\n", .{nowMs() - t0});
        return pixels;
    }
};
