// MIT License  Copyright (c) 2025 Matthew Abbott
//
// Usage example for the facaded_gan Zig bindings.
//
// Build and run:
//   cargo build --release -p facaded_gan_c   # build native lib first
//   zig build run-example

const std = @import("std");
const fg  = @import("facaded_gan");

pub fn main() !void {
    // ── Backend ──────────────────────────────────────────────────────────────
    fg.initBackend("cpu");
    std.debug.print("backend: {s}\n", .{fg.detectBackend()});

    // ── High-level run ────────────────────────────────────────────────────────
    var cfg = try fg.Config.init();
    defer cfg.deinit();

    cfg.setEpochs(2);
    cfg.setBatchSize(8);
    cfg.setNoiseDepth(64);
    cfg.setLearningRate(0.0002);

    std.debug.print("epochs={d}  batch={d}  lr={d}\n", .{
        cfg.epochs(), cfg.batchSize(), cfg.learningRate(),
    });

    var result = try fg.run(&cfg);
    defer result.deinit();

    var m = try result.metrics();
    defer m.deinit();
    std.debug.print("g_loss={d:.4}  d_real={d:.4}  d_fake={d:.4}  epoch={d}\n", .{
        m.gLoss(), m.dLossReal(), m.dLossFake(), m.epoch(),
    });

    var gen = try result.generator();
    defer gen.deinit();
    std.debug.print("generator layers: {d}\n", .{gen.layerCount()});

    // ── Manual network build ──────────────────────────────────────────────────
    const sizes = [_]c_int{ 64, 128, 1 };
    var g2 = try fg.Network.genBuild(&sizes, "leaky", "adam", 0.0002);
    defer g2.deinit();
    std.debug.print("manual gen layers: {d}\n", .{g2.layerCount()});

    // ── Matrix arithmetic ─────────────────────────────────────────────────────
    var a = try fg.Matrix.init(3, 3);
    defer a.deinit();
    a.set(0, 0, 1.0);
    a.set(1, 1, 2.0);
    a.set(2, 2, 3.0);

    var b = try a.transpose();
    defer b.deinit();
    std.debug.print("transpose[0,0]={d}  [1,1]={d}  [2,2]={d}\n", .{
        b.get(0, 0), b.get(1, 1), b.get(2, 2),
    });

    // ── Noise generation ──────────────────────────────────────────────────────
    var noise = try fg.generateNoise(8, 64, "gauss");
    defer noise.deinit();
    std.debug.print("noise shape: {d}×{d}\n", .{ noise.rows(), noise.cols() });

    // ── Security ──────────────────────────────────────────────────────────────
    std.debug.print("path safe: {}\n", .{fg.validatePath("/tmp/safe.bin")});
    std.debug.print("path safe: {}\n", .{fg.validatePath("../../etc/passwd")});
}
