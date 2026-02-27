// Background Removal via U2-Net (Salient Object Detection)
//
// This module removes the background from a generated image by running it
// through a U2-Net model, which produces a per-pixel foreground probability
// mask. The mask becomes the alpha channel in an RGBA output image.
//
// U2-Net ("U-squared Net") is a two-level nested U-Net architecture designed
// for salient object detection — identifying the most visually prominent
// object in an image. It works well for background removal because:
//
//   1. Each encoder/decoder block is itself a small U-Net (called RSU blocks),
//      giving the network multi-scale feature extraction at every level.
//   2. It produces 7 output masks at different resolutions (d0-d6). We use d0,
//      the finest/final mask that combines information from all scales.
//   3. The output already has sigmoid applied, so values are in [0, 1]:
//      1.0 = foreground (keep), 0.0 = background (transparent).
//
// The model expects 320x320 input normalized with ImageNet statistics, so we
// bilinearly resize from 512x512, normalize, run inference, then resize the
// mask back to 512x512 before compositing into RGBA.
//
// Performance: U2-Net inference on TensorRT is typically 5-15ms on an RTX GPU,
// negligible compared to the ~300ms diffusion pipeline.

const std = @import("std");
const Engine = @import("engine.zig").Engine;

/// ImageNet normalization constants.
/// These are the mean and standard deviation of the ImageNet training set.
/// U2-Net was trained on images normalized this way, so inference requires
/// the same normalization to produce correct masks.
const imagenet_mean = [3]f32{ 0.485, 0.456, 0.406 };
const imagenet_std = [3]f32{ 0.229, 0.224, 0.225 };

/// U2-Net input resolution. The model was trained on 320x320 crops.
const model_size: usize = 320;

/// Generated image dimensions (Stable Diffusion 1.5 outputs 512x512).
const image_size: usize = 512;

pub const BackgroundRemoval = struct {
    allocator: std.mem.Allocator,
    // Borrowed pointer — BackgroundRemoval does NOT own this engine.
    // Ownership belongs to InferenceModels (see inference.zig).
    engine: *Engine,

    pub fn init(allocator: std.mem.Allocator, engine: *Engine) BackgroundRemoval {
        return .{
            .allocator = allocator,
            .engine = engine,
        };
    }

    /// Remove the background from a 512x512 RGB image.
    ///
    /// Takes HWC-interleaved RGB pixels (512*512*3 bytes) and returns
    /// HWC-interleaved RGBA pixels (512*512*4 bytes) where the alpha
    /// channel is the U2-Net foreground mask.
    ///
    /// Pipeline:
    ///   1. Bilinear resize 512x512 → 320x320
    ///   2. Normalize: [0,255] → [0,1] → ImageNet mean/std
    ///   3. Convert HWC → CHW (what the model expects)
    ///   4. Run U2-Net inference → d0 mask [1, 1, 320, 320]
    ///   5. Bilinear resize mask 320x320 → 512x512
    ///   6. Composite: RGB + alpha mask → RGBA
    pub fn removeBackground(self: *BackgroundRemoval, rgb_pixels: []const u8) ![]u8 {
        const t0 = nowMs();

        // Step 1: Bilinear resize 512x512 RGB → 320x320 RGB (as floats)
        const resized = try self.allocator.alloc(f32, model_size * model_size * 3);
        defer self.allocator.free(resized);
        bilinearResize3(rgb_pixels, image_size, image_size, resized, model_size, model_size);

        // Step 2 & 3: Normalize with ImageNet stats and convert HWC → CHW
        // U2-Net input layout: [1, 3, 320, 320] = [batch, channels, height, width]
        // Each channel is a contiguous 320x320 block of normalized float values.
        const input_elements = 1 * 3 * model_size * model_size;
        const chw_input = try self.allocator.alloc(f32, input_elements);
        defer self.allocator.free(chw_input);

        for (0..model_size) |y| {
            for (0..model_size) |x| {
                for (0..3) |ch| {
                    // HWC source index: pixel at (y, x), channel ch
                    const hwc_idx = (y * model_size + x) * 3 + ch;
                    // CHW destination index: channel ch, row y, col x
                    const chw_idx = ch * model_size * model_size + y * model_size + x;

                    // Normalize: [0, 255] → [0, 1] → subtract mean → divide by std
                    const val = resized[hwc_idx] / 255.0;
                    chw_input[chw_idx] = (val - imagenet_mean[ch]) / imagenet_std[ch];
                }
            }
        }

        // Step 4: Run U2-Net inference
        // Try the common ONNX input name "input.1" first, fall back to index 0
        const inp = self.engine.findInput("input.1") orelse
            try self.engine.getInputInfo(0);

        try self.engine.setInputShape(inp.name, &[_]i64{ 1, 3, model_size, model_size });

        // Handle FP16 engines: convert float32 input to float16 if needed
        if (inp.dtype == .float16) {
            const fp16_buf = try self.allocator.alloc(u16, input_elements);
            defer self.allocator.free(fp16_buf);
            for (0..input_elements) |i| fp16_buf[i] = fp32ToFp16(chw_input[i]);
            try self.engine.setInput(inp.name, std.mem.sliceAsBytes(fp16_buf));
        } else {
            try self.engine.setInput(inp.name, std.mem.sliceAsBytes(chw_input));
        }

        try self.engine.infer();

        // Read the d0 output mask: [1, 1, 320, 320] — per-pixel foreground probability
        // Sigmoid is already applied in the model, so values are in [0, 1].
        const mask_elements = model_size * model_size;
        const mask_320 = try self.allocator.alloc(f32, mask_elements);
        defer self.allocator.free(mask_320);

        // U2-Net has 7 outputs (d0-d6). d0 is the primary output.
        // Try common ONNX output names, fall back to index 0.
        const out = self.engine.findOutput("d0") orelse
            self.engine.findOutput("1959") orelse
            try self.engine.getOutputInfo(0);

        if (out.dtype == .float16) {
            const fp16_buf = try self.allocator.alloc(u16, mask_elements);
            defer self.allocator.free(fp16_buf);
            try self.engine.getOutput(out.name, std.mem.sliceAsBytes(fp16_buf));
            for (0..mask_elements) |i| mask_320[i] = fp16ToFp32(fp16_buf[i]);
        } else {
            try self.engine.getOutput(out.name, std.mem.sliceAsBytes(mask_320));
        }

        // Normalize mask to [0, 1] range (should already be, but clamp to be safe)
        var min_val: f32 = mask_320[0];
        var max_val: f32 = mask_320[0];
        for (mask_320) |v| {
            min_val = @min(min_val, v);
            max_val = @max(max_val, v);
        }
        if (max_val > min_val) {
            const range = max_val - min_val;
            for (mask_320) |*v| {
                v.* = (v.* - min_val) / range;
            }
        }

        // Step 5: Bilinear resize mask from 320x320 → 512x512
        const mask_512 = try self.allocator.alloc(f32, image_size * image_size);
        defer self.allocator.free(mask_512);
        bilinearResize1(mask_320, model_size, model_size, mask_512, image_size, image_size);

        // Step 6: Composite RGB + alpha → RGBA
        // For each pixel, copy the original R, G, B values and add the mask as alpha.
        const rgba_pixels = try self.allocator.alloc(u8, image_size * image_size * 4);
        for (0..image_size * image_size) |i| {
            rgba_pixels[i * 4 + 0] = rgb_pixels[i * 3 + 0]; // R
            rgba_pixels[i * 4 + 1] = rgb_pixels[i * 3 + 1]; // G
            rgba_pixels[i * 4 + 2] = rgb_pixels[i * 3 + 2]; // B

            // Alpha: mask value [0, 1] → [0, 255]
            const alpha = std.math.clamp(mask_512[i], 0.0, 1.0) * 255.0;
            rgba_pixels[i * 4 + 3] = @intFromFloat(alpha + 0.5);
        }

        std.debug.print("Background removal: {d:.1} ms\n", .{nowMs() - t0});
        return rgba_pixels;
    }
};

// ── Bilinear interpolation ──────────────────────────────────────────────────
//
// Bilinear interpolation samples a source image at non-integer coordinates by
// weighting the 4 nearest pixels. For a point (sx, sy) the weights are:
//
//     (1-fx)*(1-fy) * src[y0][x0]  +  fx*(1-fy) * src[y0][x1]
//   + (1-fx)*fy     * src[y1][x0]  +  fx*fy     * src[y1][x1]
//
// where (x0, y0) is the floor and fx, fy are the fractional parts.
// This produces smooth scaling without the blocky artifacts of nearest-neighbor.

/// Bilinear resize for 3-channel (RGB) images.
/// Source is HWC u8 [src_h * src_w * 3], destination is HWC f32 [dst_h * dst_w * 3].
fn bilinearResize3(src: []const u8, src_w: usize, src_h: usize, dst: []f32, dst_w: usize, dst_h: usize) void {
    const sx_ratio = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(dst_w));
    const sy_ratio = @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(dst_h));

    for (0..dst_h) |dy| {
        for (0..dst_w) |dx| {
            // Map destination pixel to source coordinates
            const sx = @as(f32, @floatFromInt(dx)) * sx_ratio;
            const sy = @as(f32, @floatFromInt(dy)) * sy_ratio;

            // Four nearest source pixels
            const x0 = @as(usize, @intFromFloat(sx));
            const y0 = @as(usize, @intFromFloat(sy));
            const x1 = @min(x0 + 1, src_w - 1);
            const y1 = @min(y0 + 1, src_h - 1);

            // Fractional offsets for weighting
            const fx = sx - @as(f32, @floatFromInt(x0));
            const fy = sy - @as(f32, @floatFromInt(y0));

            for (0..3) |ch| {
                const p00 = @as(f32, @floatFromInt(src[(y0 * src_w + x0) * 3 + ch]));
                const p01 = @as(f32, @floatFromInt(src[(y0 * src_w + x1) * 3 + ch]));
                const p10 = @as(f32, @floatFromInt(src[(y1 * src_w + x0) * 3 + ch]));
                const p11 = @as(f32, @floatFromInt(src[(y1 * src_w + x1) * 3 + ch]));

                const val = p00 * (1 - fx) * (1 - fy) +
                    p01 * fx * (1 - fy) +
                    p10 * (1 - fx) * fy +
                    p11 * fx * fy;

                dst[(dy * dst_w + dx) * 3 + ch] = val;
            }
        }
    }
}

/// Bilinear resize for single-channel (mask) images.
/// Source is f32 [src_h * src_w], destination is f32 [dst_h * dst_w].
fn bilinearResize1(src: []const f32, src_w: usize, src_h: usize, dst: []f32, dst_w: usize, dst_h: usize) void {
    const sx_ratio = @as(f32, @floatFromInt(src_w)) / @as(f32, @floatFromInt(dst_w));
    const sy_ratio = @as(f32, @floatFromInt(src_h)) / @as(f32, @floatFromInt(dst_h));

    for (0..dst_h) |dy| {
        for (0..dst_w) |dx| {
            const sx = @as(f32, @floatFromInt(dx)) * sx_ratio;
            const sy = @as(f32, @floatFromInt(dy)) * sy_ratio;

            const x0 = @as(usize, @intFromFloat(sx));
            const y0 = @as(usize, @intFromFloat(sy));
            const x1 = @min(x0 + 1, src_w - 1);
            const y1 = @min(y0 + 1, src_h - 1);

            const fx = sx - @as(f32, @floatFromInt(x0));
            const fy = sy - @as(f32, @floatFromInt(y0));

            const p00 = src[y0 * src_w + x0];
            const p01 = src[y0 * src_w + x1];
            const p10 = src[y1 * src_w + x0];
            const p11 = src[y1 * src_w + x1];

            dst[dy * dst_w + dx] = p00 * (1 - fx) * (1 - fy) +
                p01 * fx * (1 - fy) +
                p10 * (1 - fx) * fy +
                p11 * fx * fy;
        }
    }
}

// ── FP16 ↔ FP32 conversion ─────────────────────────────────────────────────
// Same bit-manipulation approach used in pipeline.zig for TensorRT FP16 I/O.

fn fp32ToFp16(val: f32) u16 {
    const bits: u32 = @bitCast(val);
    const sign: u16 = @intCast((bits >> 16) & 0x8000);
    const exp_val = @as(i32, @intCast((bits >> 23) & 0xFF)) - 127 + 15;
    const frac: u16 = @intCast((bits >> 13) & 0x3FF);
    if (exp_val <= 0) return sign;
    if (exp_val >= 31) return sign | 0x7C00;
    return sign | (@as(u16, @intCast(exp_val)) << 10) | frac;
}

fn fp16ToFp32(val: u16) f32 {
    const sign: u32 = @as(u32, val & 0x8000) << 16;
    const exp_val: u32 = (val >> 10) & 0x1F;
    const frac: u32 = @as(u32, val & 0x03FF);
    if (exp_val == 0) return @bitCast(sign);
    if (exp_val == 31) return @bitCast(sign | 0x7F800000 | (frac << 13));
    const biased = (exp_val + 127 - 15) << 23;
    return @bitCast(sign | biased | (frac << 13));
}

fn nowMs() f64 {
    const ts = std.time.Instant.now() catch return 0;
    return @as(f64, @floatFromInt(ts.timestamp.sec)) * 1000.0 +
        @as(f64, @floatFromInt(ts.timestamp.nsec)) / 1_000_000.0;
}
