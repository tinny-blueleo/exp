const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Library module ──────────────────────────────────────────────────
    // Public module "sd_tensorrt" rooted at src/lib.zig.
    // External Zig projects can depend on this via build.zig.zon.
    const sd_tensorrt = b.addModule("sd_tensorrt", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    // Include paths: C++ wrapper headers + system TensorRT/CUDA headers.
    sd_tensorrt.addIncludePath(b.path("trt_wrapper"));
    sd_tensorrt.addIncludePath(b.path("include"));
    sd_tensorrt.addSystemIncludePath(.{ .cwd_relative = "/usr/include/x86_64-linux-gnu" });
    sd_tensorrt.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });

    // Library search paths.
    sd_tensorrt.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    // Link TensorRT and CUDA runtime.
    sd_tensorrt.linkSystemLibrary("nvinfer", .{});
    sd_tensorrt.linkSystemLibrary("cudart", .{});

    // Compile C++ wrapper files (TensorRT engine + C bridge).
    const cpp_flags: []const []const u8 = &.{"-std=c++17"};
    sd_tensorrt.addCSourceFile(.{ .file = b.path("trt_wrapper/trt_engine.cpp"), .flags = cpp_flags });
    sd_tensorrt.addCSourceFile(.{ .file = b.path("trt_wrapper/trt_wrapper.cpp"), .flags = cpp_flags });

    // Compile stb_image_write implementation (as C++ for default arg support).
    sd_tensorrt.addCSourceFile(.{ .file = b.path("trt_wrapper/stb_impl.cpp"), .flags = cpp_flags });

    // ── CLI executable ──────────────────────────────────────────────────
    // Imports the sd_tensorrt library module. Handles CLI arg parsing and PNG output.
    const exe = b.addExecutable(.{
        .name = "sd-tensorrt-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "sd_tensorrt", .module = sd_tensorrt },
            },
        }),
    });

    b.installArtifact(exe);

    // `zig build run -- --prompt "..." --seed 42`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the stable diffusion pipeline");
    run_step.dependOn(&run_cmd.step);
}
