// Inference model manager — loads all TensorRT engines once for reuse.
//
// In a server context, you'd call InferenceModels.init() at startup to load
// all engines into GPU memory, then call generate() / generateTransparent()
// for each incoming request. The engines stay loaded for the lifetime of the
// server, avoiding the multi-second cost of reloading per request.
//
// Ownership model:
//   InferenceModels owns all engines, the tokenizer, and the scheduler.
//   Pipeline and BackgroundRemoval borrow pointers to these — they don't
//   create or destroy any engines themselves.
//
// TODO: For concurrent server use, generate() and generateTransparent() are
// NOT thread-safe. TensorRT execution contexts hold internal GPU state that
// corrupts under concurrent access. Two options:
//   1. Wrap calls in a std.Thread.Mutex (simple, limits to 1 request at a time)
//   2. Create an engine pool with N copies of each engine for N-way concurrency
//      (higher throughput, but multiplies GPU memory usage by N)
//
// TODO: Port TensorRT builder API to Zig (via C wrapper) so init() can convert
// ONNX → TRT engines on first startup. This eliminates the Python dependency
// for deployment. TRT engines are NOT portable across GPU architectures, so
// the conversion must happen on the target server's GPU. The flow would be:
//   init → check for cached .trt files → if missing, build from ONNX → load
// Currently this step requires running the Python scripts in scripts/.

const std = @import("std");
const eng = @import("../engine.zig");
const Pipeline = @import("pipeline.zig").Pipeline;
const BackgroundRemoval = @import("background_removal.zig").BackgroundRemoval;
const ClipTokenizer = @import("tokenizer.zig").ClipTokenizer;
const LcmScheduler = @import("scheduler.zig").LcmScheduler;

const Engine = eng.Engine;

pub const ModelConfig = struct {
    engine_dir: []const u8,
    vocab_path: []const u8,
    enable_transparency: bool,
};

pub const InferenceModels = struct {
    allocator: std.mem.Allocator,

    // Core diffusion engines (always loaded)
    text_encoder: Engine,
    unet: Engine,
    vae_decoder: Engine,

    // Background removal engine (only loaded if transparency enabled)
    u2net: ?Engine,

    // Tokenizer and scheduler (pure CPU, no GPU resources)
    tokenizer: ClipTokenizer,
    scheduler: LcmScheduler,

    // Engine paths needed for sequential mode reload
    text_encoder_path: [*:0]const u8,
    unet_path: [*:0]const u8,
    vae_decoder_path: [*:0]const u8,

    sequential_mode: bool,

    /// Load all TensorRT engines from the given engine directory.
    /// This is the expensive step (~2-3 seconds) that should happen once at startup.
    /// After this, generate() and generateTransparent() are fast (~450ms each).
    pub fn init(allocator: std.mem.Allocator, config: ModelConfig) !InferenceModels {
        const text_enc_path = try std.fmt.allocPrintSentinel(allocator, "{s}/text_encoder.trt", .{config.engine_dir}, 0);
        errdefer allocator.free(text_enc_path);
        const unet_path = try std.fmt.allocPrintSentinel(allocator, "{s}/unet.trt", .{config.engine_dir}, 0);
        errdefer allocator.free(unet_path);
        const vae_path = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.trt", .{config.engine_dir}, 0);
        errdefer allocator.free(vae_path);

        var self = InferenceModels{
            .allocator = allocator,
            .text_encoder = try Engine.init(),
            .unet = try Engine.init(),
            .vae_decoder = try Engine.init(),
            .u2net = null,
            .tokenizer = ClipTokenizer.init(allocator),
            .scheduler = .{},
            .text_encoder_path = text_enc_path,
            .unet_path = unet_path,
            .vae_decoder_path = vae_path,
            .sequential_mode = false,
        };

        // Load tokenizer (CPU-only, reads BPE vocab file)
        try self.tokenizer.load(config.vocab_path);

        // Initialize scheduler (pure math, no I/O)
        self.scheduler.init();

        // Load the three core diffusion engines into GPU memory.
        // If VRAM is insufficient for all three simultaneously, fall back to
        // sequential mode where only one engine is loaded at a time.
        try self.text_encoder.load(text_enc_path);
        try self.unet.load(unet_path);

        self.vae_decoder.load(vae_path) catch {
            std.debug.print("VRAM limited — switching to sequential engine loading\n", .{});
            self.sequential_mode = true;
            self.unet.unload();
            self.text_encoder.unload();
        };

        // Optionally load U2-Net for background removal
        if (config.enable_transparency) {
            const u2net_path = try std.fmt.allocPrintSentinel(allocator, "{s}/u2net.trt", .{config.engine_dir}, 0);
            // u2net_path is not stored separately; the Engine holds onto it after load.
            // We don't defer free here because on error, deinit handles cleanup.
            var u2net_engine = try Engine.init();
            try u2net_engine.load(u2net_path);
            self.u2net = u2net_engine;
            std.debug.print("Loaded U2-Net engine for background removal\n", .{});
            allocator.free(u2net_path);
        }

        if (self.sequential_mode) {
            std.debug.print("All models initialized (sequential mode)\n", .{});
        } else {
            std.debug.print("All models initialized\n", .{});
        }

        return self;
    }

    pub fn deinit(self: *InferenceModels) void {
        if (self.u2net) |*u| u.deinit();
        self.vae_decoder.deinit();
        self.unet.deinit();
        self.text_encoder.deinit();
        self.tokenizer.deinit();
        // Convert sentinel pointers back to slices for freeing
        self.allocator.free(std.mem.span(self.text_encoder_path));
        self.allocator.free(std.mem.span(self.unet_path));
        self.allocator.free(std.mem.span(self.vae_decoder_path));
    }

    /// Generate a 512x512 RGB image from a text prompt.
    /// Returns caller-owned pixel buffer (512*512*3 bytes, HWC interleaved).
    pub fn generate(self: *InferenceModels, prompt: []const u8, seed: u32, steps: i32) ![]u8 {
        var pipeline = Pipeline.init(
            self.allocator,
            &self.text_encoder,
            &self.unet,
            &self.vae_decoder,
            &self.tokenizer,
            &self.scheduler,
            self.text_encoder_path,
            self.unet_path,
            self.vae_decoder_path,
            self.sequential_mode,
        );
        return pipeline.generate(prompt, seed, steps);
    }

    /// Generate a 512x512 RGBA image with the background removed.
    /// Returns caller-owned pixel buffer (512*512*4 bytes, HWC interleaved).
    /// Requires enable_transparency=true in ModelConfig.
    pub fn generateTransparent(self: *InferenceModels, prompt: []const u8, seed: u32, steps: i32) ![]u8 {
        const rgb_pixels = try self.generate(prompt, seed, steps);
        defer self.allocator.free(rgb_pixels);

        const u2net_ptr = &(self.u2net orelse return error.U2NetNotLoaded);
        var bg_removal = BackgroundRemoval.init(self.allocator, u2net_ptr);
        return bg_removal.removeBackground(rgb_pixels);
    }
};
