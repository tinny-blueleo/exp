// sd_tensorrt — GPU-accelerated inference library for image/audio generation.
//
// This is the library root. External projects import this module to access
// inference pipelines. Each pipeline lives in its own subdirectory.
//
// Usage from another Zig project:
//
//   const sd = @import("sd_tensorrt");
//
//   var models = try sd.TextToImageModels.init(allocator, .{
//       .engine_dir = "engines",
//       .vocab_path = "data/bpe_simple_vocab_16e6.txt",
//       .enable_transparency = true,
//   });
//   defer models.deinit();
//
//   const pixels = try sd.generateImageForPrompt(&models, .{
//       .prompt = "ninja cat",
//       .seed = 42,
//       .transparent = true,
//   });

pub const text_to_image = @import("text_to_image/inference.zig");

// Re-export key types at the top level for convenience.
pub const TextToImageModels = text_to_image.InferenceModels;
pub const TextToImageConfig = text_to_image.ModelConfig;

/// Parameters for text-to-image generation.
pub const ImageFromPromptParams = struct {
    prompt: []const u8,
    seed: u32 = 42,
    steps: i32 = 4,
    transparent: bool = false,
};

/// Generate a 512x512 image from a text prompt.
///
/// Returns a caller-owned pixel buffer:
///   - RGB  (512*512*3 bytes) if transparent=false
///   - RGBA (512*512*4 bytes) if transparent=true
///
/// The models must have been initialized with enable_transparency=true
/// if transparent=true is requested.
pub fn generateImageForPrompt(models: *TextToImageModels, params: ImageFromPromptParams) ![]u8 {
    if (params.transparent) {
        return models.generateTransparent(params.prompt, params.seed, params.steps);
    } else {
        return models.generate(params.prompt, params.seed, params.steps);
    }
}
