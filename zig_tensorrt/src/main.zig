// Zig TensorRT Stable Diffusion — CLI entry point.
//
// Generates a 512x512 PNG image from a text prompt using LCM Dreamshaper v7.
// Uses the same pre-built TensorRT engines as the C++ version.
//
// Usage:
//   zig build run -- --prompt "Taj Mahal in space" --seed 42 --steps 4 -o output.png

const std = @import("std");
const Pipeline = @import("pipeline.zig").Pipeline;
const PipelineConfig = @import("pipeline.zig").PipelineConfig;

// stb_image_write — declare the function directly since the header uses C++
// default arguments (= NULL) that Zig's C translator can't parse.
extern fn stbi_write_png(filename: [*:0]const u8, w: c_int, h: c_int, comp: c_int, data: *const anyopaque, stride_in_bytes: c_int, parameters: ?[*:0]const u8) c_int;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Parse CLI arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var prompt: []const u8 = "Taj Mahal in space";
    var seed: u32 = 42;
    var steps: i32 = 4;
    var output: [:0]const u8 = "output.png";
    var engine_dir: []const u8 = "engines";
    var vocab: []const u8 = "data/bpe_simple_vocab_16e6.txt";

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--prompt") and i + 1 < args.len) {
            i += 1;
            prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--seed") and i + 1 < args.len) {
            i += 1;
            seed = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--steps") and i + 1 < args.len) {
            i += 1;
            steps = try std.fmt.parseInt(i32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "-o") and i + 1 < args.len) {
            i += 1;
            output = args[i];
        } else if (std.mem.eql(u8, arg, "--engine-dir") and i + 1 < args.len) {
            i += 1;
            engine_dir = args[i];
        } else if (std.mem.eql(u8, arg, "--vocab") and i + 1 < args.len) {
            i += 1;
            vocab = args[i];
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            std.debug.print(
                \\Usage: sd-tensorrt-zig [options]
                \\  --prompt TEXT   Text prompt (default: "Taj Mahal in space")
                \\  --seed N        Random seed (default: 42)
                \\  --steps N       LCM denoising steps (default: 4)
                \\  -o FILE         Output PNG path (default: output.png)
                \\  --engine-dir D  Engine directory (default: engines/)
                \\  --vocab FILE    BPE vocab file (default: data/bpe_simple_vocab_16e6.txt)
                \\  -h, --help      Show this help
                \\
            , .{});
            return;
        } else {
            std.debug.print("Unknown option: {s}\n", .{arg});
            return error.InvalidArgument;
        }
    }

    // Build engine paths
    const text_enc_path = try std.fmt.allocPrintSentinel(allocator, "{s}/text_encoder.trt", .{engine_dir}, 0);
    defer allocator.free(text_enc_path);
    const unet_path = try std.fmt.allocPrintSentinel(allocator, "{s}/unet.trt", .{engine_dir}, 0);
    defer allocator.free(unet_path);
    const vae_path = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.trt", .{engine_dir}, 0);
    defer allocator.free(vae_path);

    const config = PipelineConfig{
        .text_encoder_engine = text_enc_path,
        .unet_engine = unet_path,
        .vae_decoder_engine = vae_path,
        .vocab_path = vocab,
    };

    // Initialize: loads tokenizer, scheduler, and all TensorRT engines
    var pipeline = try Pipeline.init(allocator, config);
    defer pipeline.deinit();

    // Run the full pipeline: text encoding → UNet denoising → VAE decode
    const pixels = try pipeline.generate(prompt, seed, steps);
    defer allocator.free(pixels);

    // Write the RGB pixel data as a PNG file
    const ok = stbi_write_png(output.ptr, 512, 512, 3, pixels.ptr, 512 * 3, null);
    if (ok != 0) {
        std.debug.print("Saved: {s}\n", .{output});
    } else {
        std.debug.print("Failed to write: {s}\n", .{output});
        return error.WriteFailed;
    }
}
