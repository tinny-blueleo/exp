// Zig wrapper around the TensorRT C API (trt_wrapper.h).
//
// TensorRT is a C++ library with virtual classes — Zig can't call C++ directly.
// Instead, trt_wrapper.h exposes a flat C API with opaque TrtEngine* handles.
// This module wraps those C functions into idiomatic Zig with error returns.
const std = @import("std");

const c = @cImport({
    @cInclude("trt_wrapper.h");
});

/// Mirrors nvinfer1::DataType. Determines how many bytes each tensor element uses.
pub const DataType = enum(i32) {
    float32 = 0,
    float16 = 1,
    int8 = 2,
    int32 = 3,
    bool_ = 4,
    uint8 = 5,
    fp8 = 6,
    bf16 = 7,
    int64 = 8,

    pub fn byteSize(self: DataType) usize {
        return switch (self) {
            .float32, .int32 => 4,
            .float16, .bf16 => 2,
            .int8, .bool_, .uint8, .fp8 => 1,
            .int64 => 8,
        };
    }
};

/// Metadata for one I/O tensor: its name, data type, shape, and GPU buffer size.
pub const TensorInfo = struct {
    name: [:0]const u8,
    dtype: DataType,
    dims: [8]i64,
    num_dims: i32,
    byte_size: usize,
};

pub const EngineError = error{
    CreateFailed,
    LoadFailed,
    SetShapeFailed,
    SetInputFailed,
    InferenceFailed,
    GetOutputFailed,
    TensorNotFound,
};

/// A TensorRT engine loaded into GPU memory, ready for inference.
pub const Engine = struct {
    handle: *c.TrtEngine,

    pub fn init() EngineError!Engine {
        const h = c.trt_engine_create() orelse return EngineError.CreateFailed;
        return .{ .handle = h };
    }

    pub fn deinit(self: *Engine) void {
        c.trt_engine_destroy(self.handle);
    }

    pub fn load(self: *Engine, path: [*:0]const u8) EngineError!void {
        if (!c.trt_engine_load(self.handle, path)) {
            return EngineError.LoadFailed;
        }
    }

    pub fn unload(self: *Engine) void {
        c.trt_engine_unload(self.handle);
    }

    pub fn isLoaded(self: *const Engine) bool {
        return c.trt_engine_is_loaded(self.handle);
    }

    pub fn numInputs(self: *const Engine) usize {
        return @intCast(c.trt_engine_num_inputs(self.handle));
    }

    pub fn numOutputs(self: *const Engine) usize {
        return @intCast(c.trt_engine_num_outputs(self.handle));
    }

    pub fn getInputInfo(self: *const Engine, index: usize) EngineError!TensorInfo {
        var ci: c.TrtTensorInfo = undefined;
        if (!c.trt_engine_get_input_info(self.handle, @intCast(index), &ci))
            return EngineError.TensorNotFound;
        return fromCInfo(ci);
    }

    pub fn getOutputInfo(self: *const Engine, index: usize) EngineError!TensorInfo {
        var ci: c.TrtTensorInfo = undefined;
        if (!c.trt_engine_get_output_info(self.handle, @intCast(index), &ci))
            return EngineError.TensorNotFound;
        return fromCInfo(ci);
    }

    pub fn findInput(self: *const Engine, name: [*:0]const u8) ?TensorInfo {
        var ci: c.TrtTensorInfo = undefined;
        if (!c.trt_engine_find_input(self.handle, name, &ci)) return null;
        return fromCInfo(ci);
    }

    pub fn findOutput(self: *const Engine, name: [*:0]const u8) ?TensorInfo {
        var ci: c.TrtTensorInfo = undefined;
        if (!c.trt_engine_find_output(self.handle, name, &ci)) return null;
        return fromCInfo(ci);
    }

    pub fn setInputShape(self: *Engine, name: [*:0]const u8, dims: []const i64) EngineError!void {
        if (!c.trt_engine_set_input_shape(self.handle, name, dims.ptr, @intCast(dims.len)))
            return EngineError.SetShapeFailed;
    }

    pub fn setInput(self: *Engine, name: [*:0]const u8, data: []const u8) EngineError!void {
        if (!c.trt_engine_set_input(self.handle, name, data.ptr, data.len))
            return EngineError.SetInputFailed;
    }

    pub fn infer(self: *Engine) EngineError!void {
        if (!c.trt_engine_infer(self.handle))
            return EngineError.InferenceFailed;
    }

    pub fn getOutput(self: *Engine, name: [*:0]const u8, buf: []u8) EngineError!void {
        if (!c.trt_engine_get_output(self.handle, name, buf.ptr, buf.len))
            return EngineError.GetOutputFailed;
    }

    fn fromCInfo(ci: c.TrtTensorInfo) TensorInfo {
        return .{
            .name = std.mem.span(ci.name),
            .dtype = @enumFromInt(ci.dtype),
            .dims = ci.dims,
            .num_dims = ci.num_dims,
            .byte_size = ci.byte_size,
        };
    }
};
