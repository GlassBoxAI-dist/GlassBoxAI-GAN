// MIT License  Copyright (c) 2025 Matthew Abbott
//
// Build script for the facaded_gan Zig bindings.
//
// PREREQUISITES
// -------------
// Build the native library first:
//   cargo build --release -p facaded_gan_c
//
// Then build this package:
//   zig build            (builds static lib + example)
//   zig build run-example

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target   = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_dir = b.path("../target/release");

    // ── Reusable module ────────────────────────────────────────────────────
    const mod = b.createModule(.{
        .root_source_file = b.path("src/facaded_gan.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    mod.addLibraryPath(lib_dir);
    mod.linkSystemLibrary("facaded_gan_c", .{});
    mod.link_libc = true;

    // ── Static library (for consumers that link directly) ──────────────────
    const lib = b.addLibrary(.{
        .name        = "facaded_gan_zig",
        .root_module = mod,
        .linkage     = .static,
    });
    b.installArtifact(lib);

    // ── Example executable ─────────────────────────────────────────────────
    const ex_mod = b.createModule(.{
        .root_source_file = b.path("src/example.zig"),
        .target           = target,
        .optimize         = optimize,
    });
    ex_mod.addImport("facaded_gan", mod);
    ex_mod.addLibraryPath(lib_dir);
    ex_mod.linkSystemLibrary("facaded_gan_c", .{});
    ex_mod.link_libc = true;

    const example  = b.addExecutable(.{
        .name        = "example",
        .root_module = ex_mod,
    });
    b.installArtifact(example);

    const run_cmd  = b.addRunArtifact(example);
    const run_step = b.step("run-example", "Run the usage example");
    run_step.dependOn(&run_cmd.step);
}
