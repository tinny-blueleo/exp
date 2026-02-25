const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the root module (Zig 0.16 requires this before addExecutable).
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    // Include paths: our headers + system TensorRT/CUDA headers.
    root_mod.addIncludePath(b.path("src"));
    root_mod.addIncludePath(b.path("include"));
    root_mod.addSystemIncludePath(.{ .cwd_relative = "/usr/include/x86_64-linux-gnu" });
    root_mod.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });

    // Library search paths.
    root_mod.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });

    // Link TensorRT and CUDA runtime.
    root_mod.linkSystemLibrary("nvinfer", .{});
    root_mod.linkSystemLibrary("cudart", .{});

    // Compile C++ wrapper files (TensorRT engine + C bridge).
    const cpp_flags: []const []const u8 = &.{"-std=c++17"};
    root_mod.addCSourceFile(.{ .file = b.path("src/trt_engine.cpp"), .flags = cpp_flags });
    root_mod.addCSourceFile(.{ .file = b.path("src/trt_wrapper.cpp"), .flags = cpp_flags });

    // Compile stb_image_write implementation (as C++ for default arg support).
    root_mod.addCSourceFile(.{ .file = b.path("src/stb_impl.cpp"), .flags = cpp_flags });

    const exe = b.addExecutable(.{
        .name = "sd-tensorrt-zig",
        .root_module = root_mod,
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
