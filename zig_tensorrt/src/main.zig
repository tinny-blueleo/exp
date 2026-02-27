// Zig TensorRT Stable Diffusion — CLI entry point.
//
// Generates a 512x512 PNG image from a text prompt using LCM Dreamshaper v7.
// Uses pre-built TensorRT engines loaded once via InferenceModels.
//
// Optionally removes the background using a U2-Net model (--transparent flag),
// producing an RGBA PNG with the foreground preserved and background transparent.
//
// Usage:
//   zig build run -- --prompt "Taj Mahal in space" --seed 42 --steps 4 -o output.png
//   zig build run -- --prompt "a cat" --transparent -o cat_transparent.png

const std = @import("std");
const InferenceModels = @import("inference.zig").InferenceModels;
const ModelConfig = @import("inference.zig").ModelConfig;

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
    var transparent: bool = false;

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
        } else if (std.mem.eql(u8, arg, "--transparent")) {
            transparent = true;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            std.debug.print(
                \\Usage: sd-tensorrt-zig [options]
                \\  --prompt TEXT       Text prompt (default: "Taj Mahal in space")
                \\  --seed N           Random seed (default: 42)
                \\  --steps N          LCM denoising steps (default: 4)
                \\  -o FILE            Output PNG path (default: output.png)
                \\  --engine-dir D     Engine directory (default: engines/)
                \\  --vocab FILE       BPE vocab file (default: data/bpe_simple_vocab_16e6.txt)
                \\  --transparent      Remove background (requires U2-Net engine in engine-dir)
                \\  -h, --help         Show this help
                \\
            , .{});
            return;
        } else {
            std.debug.print("Unknown option: {s}\n", .{arg});
            return error.InvalidArgument;
        }
    }

    // Load all models once — this is the expensive step.
    // In a server, you'd do this at startup and reuse `models` for every request.
    var models = try InferenceModels.init(allocator, .{
        .engine_dir = engine_dir,
        .vocab_path = vocab,
        .enable_transparency = transparent,
    });
    defer models.deinit();

    if (transparent) {
        // Generate RGBA image with background removed
        const rgba_pixels = try models.generateTransparent(prompt, seed, steps);
        defer allocator.free(rgba_pixels);

        const ok = stbi_write_png(output.ptr, 512, 512, 4, rgba_pixels.ptr, 512 * 4, null);
        if (ok != 0) {
            std.debug.print("Saved (RGBA): {s}\n", .{output});
        } else {
            std.debug.print("Failed to write: {s}\n", .{output});
            return error.WriteFailed;
        }
    } else {
        // Generate standard RGB image
        const rgb_pixels = try models.generate(prompt, seed, steps);
        defer allocator.free(rgb_pixels);

        const ok = stbi_write_png(output.ptr, 512, 512, 3, rgb_pixels.ptr, 512 * 3, null);
        if (ok != 0) {
            std.debug.print("Saved: {s}\n", .{output});
        } else {
            std.debug.print("Failed to write: {s}\n", .{output});
            return error.WriteFailed;
        }
    }
}
