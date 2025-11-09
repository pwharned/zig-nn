// src/main.zig
const std = @import("std");
//

// Two-Layer Neural Network Weight Matrix Dimensions
//
// INPUT LAYER          HIDDEN LAYER           OUTPUT LAYER
//    (784)      →W1→      (128)       →W2→       (10)
//
//
// Sample Data:
// ┌─────────┐
// │ x₁      │  784 features
// │ x₂      │  (pixels)
// │  ⋮      │
// │ x₇₈₄   │
// └─────────┘
//   60000 samples
//
//
// LAYER 1: Input → Hidden
// ─────────────────────────
//
//     (60000 × 784)    ×    (784 × 128)    =    (60000 × 128)
//     ┌──────────┐         ┌─────┐              ┌─────┐
//     │          │         │     │              │     │
//     │  X_train │    ×    │ W1  │      =       │ Z1  │
//     │          │         │     │              │     │
//     └──────────┘         └─────┘              └─────┘
//       M × K                K × N                M × N
//
// Then apply: ReLU(Z1) → A1 (still 60000 × 128)
//
//
// LAYER 2: Hidden → Output
// ─────────────────────────
//
//     (60000 × 128)    ×    (128 × 10)     =    (60000 × 10)
//     ┌─────┐              ┌──┐                 ┌──┐
//     │     │              │  │                 │  │
//     │ A1  │         ×    │W2│         =       │Z2│
//     │     │              │  │                 │  │
//     └─────┘              └──┘                 └──┘
//       M × K               K × N                M × N
//
// Then apply: Softmax(Z2) → Predictions (still 60000 × 10)

//// OpenBLAS C API bindings
extern "c" fn cblas_dgemm(
    order: c_int,
    transA: c_int,
    transB: c_int,
    M: c_int,
    N: c_int,
    K: c_int,
    alpha: f64,
    A: [*]const f64,
    lda: c_int,
    B: [*]const f64,
    ldb: c_int,
    beta: f64,
    C: [*]f64,
    ldc: c_int,
) void;

extern "c" fn cblas_daxpy(
    n: c_int,
    alpha: f64,
    X: [*]const f64,
    incX: c_int,
    Y: [*]f64,
    incY: c_int,
) void;
fn relu(a: []f64, b: []f64) void {
    for (a, 0..) |*val, idx| {
        b[idx] = @max(0.0, val.*);
    }
}
fn add_bias(z: []f64, bias: []const f64, m: usize, n_classes: usize) void {
    for (0..m) |i| {
        for (0..n_classes) |j| {
            z[i * n_classes + j] += bias[j];
        }
    }
}

// Usage:

// OpenBLAS threading functions
extern "c" fn openblas_set_num_threads(num_threads: c_int) void;
extern "c" fn openblas_get_num_threads() c_int;
extern "c" fn openblas_get_config() [*:0]const u8;

const CblasRowMajor: c_int = 101;
const CblasNoTrans: c_int = 111;
const CblasTrans: c_int = 112; // Add this!
fn get_predictions(a2: []const f64, predictions: []usize, m: usize, n_classes: usize) void {
    for (0..m) |i| {
        const row = a2[i * n_classes .. (i + 1) * n_classes];

        // Find index of maximum value in this row
        var max_idx: usize = 0;
        var max_val = row[0];
        for (row[1..], 1..) |val, j| {
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        predictions[i] = max_idx;
    }
}
fn softmax(z: []const f64, a: []f64, m: usize, n_classes: usize) void {
    for (0..m) |i| {
        const z_row = z[i * n_classes .. (i + 1) * n_classes];
        const a_row = a[i * n_classes .. (i + 1) * n_classes];

        // Step 1: Find max for numerical stability
        var max_val = z_row[0];
        for (z_row[1..]) |val| {
            max_val = @max(max_val, val);
        }

        // Step 2: Compute exp(z - max) and sum
        var sum: f64 = 0.0;
        for (z_row, 0..) |val, j| {
            a_row[j] = @exp(val - max_val);
            sum += a_row[j];
        }

        // Step 3: Normalize by sum
        for (a_row) |*val| {
            val.* /= sum;
        }
    }
}
fn get_accuracy(predictions: []const usize, y_train: []const f64, m: usize) f64 {
    var correct: usize = 0;

    for (0..m) |i| {
        const true_label = @as(usize, @intFromFloat(y_train[i]));
        if (predictions[i] == true_label) {
            correct += 1;
        }
    }

    const accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(m));
    return accuracy * 100.0; // Return as percentage
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const args = try std.process.argsAlloc(allocator);
    if (args.len < 2) {
        @panic("Provide the path to the training data");
    }
    var num_epochs: usize = 100;
    if (args.len > 2) {
        num_epochs = try std.fmt.parseInt(usize, args[2], 10);
    }

    const stdout = std.io.getStdOut().writer();
    var alpha: f64 = 0.01; // Learning rate
    if (args.len > 3) {
        alpha = try std.fmt.parseFloat(f64, args[3]);
    }

    try stdout.print("Training rate: {any}\n", .{alpha});
    defer std.process.argsFree(allocator, args);
    const file_path = args[1];
    const data = try std.fs.cwd().readFileAlloc(allocator, file_path, std.math.maxInt(usize));
    defer allocator.free(data);

    // Set number of threads (use all available cores)
    const num_threads = @as(c_int, @intCast(try std.Thread.getCpuCount()));
    openblas_set_num_threads(num_threads);

    try stdout.print("OpenBLAS config: {s}\n", .{openblas_get_config()});
    try stdout.print("OpenBLAS threads: {d}\n\n", .{openblas_get_num_threads()});

    // Large matrices
    const m: usize = 60000;
    const k: usize = 784;
    const n_hidden: usize = 128;
    const n_classes = 10;
    // Target
    const Y_train = try allocator.alloc(f64, m);
    defer allocator.free(Y_train);

    const predictions = try allocator.alloc(usize, m);
    defer allocator.free(predictions);
    const W1 = try allocator.alloc(f64, n_hidden * k);
    defer allocator.free(W1);
    const b1 = try allocator.alloc(f64, n_hidden);
    defer allocator.free(b1);
    const b2 = try allocator.alloc(f64, n_classes);
    defer allocator.free(b2);

    const W2 = try allocator.alloc(f64, n_hidden * n_classes);
    defer allocator.free(W2);
    const Z1 = try allocator.alloc(f64, n_hidden * m);
    defer allocator.free(Z1);
    const A1 = try allocator.alloc(f64, n_hidden * m);
    defer allocator.free(A1);
    const Z2 = try allocator.alloc(f64, n_classes * m);
    defer allocator.free(Z2);

    const X_train = try allocator.alloc(f64, m * k);
    defer allocator.free(X_train);
    // Gradients (same shapes as their corresponding forward pass matrices)
    const A2 = try allocator.alloc(f64, m * n_classes); // (60000 × 10)
    defer allocator.free(A2);
    @memset(A2, 0.0);
    // Layer 2 gradients
    const dZ2 = try allocator.alloc(f64, m * n_classes); // (60000 × 10)
    defer allocator.free(dZ2);

    const dW2 = try allocator.alloc(f64, n_hidden * n_classes); // (128 × 10)
    defer allocator.free(dW2);

    const db2 = try allocator.alloc(f64, n_classes); // (10)
    defer allocator.free(db2);

    // Layer 1 gradients
    const dZ1 = try allocator.alloc(f64, m * n_hidden); // (60000 × 128)
    defer allocator.free(dZ1);

    const dW1 = try allocator.alloc(f64, k * n_hidden); // (784 × 128)
    defer allocator.free(dW1);

    const db1 = try allocator.alloc(f64, n_hidden); // (128)
    defer allocator.free(db1);

    // Helper: One-hot encoded labels
    const one_hot_Y = try allocator.alloc(f64, m * n_classes); // (60000 × 10)
    // zig
    // Convert Y_train (class labels) to one-hot encoding
    defer allocator.free(one_hot_Y);

    try stdout.print("Initializing matrices...\n", .{});

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = prng.random();
    var string: [10]u8 = undefined;

    var len: usize = 0;
    var row: usize = 0;
    var col: usize = 0;
    for (data) |c| {
        if ((c == ',' or c == '\n') and len > 0) {
            if (col == 0) {
                Y_train[row] = try std.fmt.parseFloat(f64, string[0..len]);
            } else {
                X_train[row * k + (col - 1)] = try std.fmt.parseFloat(f64, string[0..len]) / 255.0;
            }
            if (c == '\n') {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }

            len = 0;
        } else {
            string[len] = c;
            len += 1;
        }
    }
    @memset(one_hot_Y, 0.0);
    for (0..m) |i| {
        const label = @as(usize, @intFromFloat(Y_train[i]));
        one_hot_Y[i * n_classes + label] = 1.0;
    }

    for (W1) |*val| {
        val.* = random.float(f64) * 2.0 - 1.0;
    }
    for (W2) |*val| {
        val.* = random.float(f64) * 2.0 - 1.0;
    }
    for (b1) |*val| {
        val.* = random.float(f64) * 2.0 - 1.0;
    }
    for (b2) |*val| {
        val.* = random.float(f64) * 2.0 - 1.0;
    }

    @memset(Z1, 0.0);

    try stdout.print("Computing C = A * B...\n", .{});

    const start = std.time.nanoTimestamp();
    var epoch: usize = 0;
    while (epoch < num_epochs) {
        epoch += 1;
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            @intCast(m),
            @intCast(n_hidden),
            @intCast(k),
            1.0,
            X_train.ptr,
            @intCast(k),
            W1.ptr,
            @intCast(n_hidden),
            0.0,
            Z1.ptr,
            @intCast(n_hidden),
        );
        // Relu

        add_bias(Z1, b1, m, n_hidden);
        relu(Z1, A1);
        // Z2 = A1 × W2
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            @intCast(m), // 60000
            @intCast(n_classes), // 10
            @intCast(n_hidden), // 128
            1.0,
            A1.ptr, // FIRST: (60000 × 128) ✓
            @intCast(n_hidden), // lda = 128 ✓
            W2.ptr, // SECOND: (128 × 10) ✓
            @intCast(n_classes), // ldb = 10 ✓
            0.0,
            Z2.ptr, // (60000 × 10) ✓
            @intCast(n_classes), // ldc = 10 ✓
        );
        add_bias(Z2, b2, m, n_classes);
        softmax(Z2, A2, m, n_classes);
        // ============================================
        // BACKWARD PASS
        // ============================================

        // Step 1: Compute dZ2 = A2 - one_hot_Y
        // Element-wise subtraction
        for (0..m * n_classes) |i| {
            dZ2[i] = A2[i] - one_hot_Y[i];
        }

        // Step 2: Compute dW2 = (1/m) * A1^T × dZ2
        // Result: (128 × 10)
        const m_inv: f64 = 1.0 / @as(f64, @floatFromInt(m));
        cblas_dgemm(
            CblasRowMajor,
            CblasTrans, // Transpose A1
            CblasNoTrans,
            @intCast(n_hidden), // M = 128 (rows in A1^T)
            @intCast(n_classes), // N = 10 (cols in dZ2)
            @intCast(m), // K = 60000 (inner dimension)
            m_inv, // alpha = 1/m
            A1.ptr, // (60000 × 128), used transposed
            @intCast(n_hidden), // lda = 128 (cols in stored A1)
            dZ2.ptr, // (60000 × 10)
            @intCast(n_classes), // ldb = 10
            0.0,
            dW2.ptr, // (128 × 10)
            @intCast(n_classes), // ldc = 10
        );

        // Step 3: Compute db2 = (1/m) * sum(dZ2, axis=0)
        // Sum each column across all samples
        @memset(db2, 0.0);
        for (0..m) |i| {
            for (0..n_classes) |j| {
                db2[j] += dZ2[i * n_classes + j];
            }
        }
        for (db2) |*val| {
            val.* *= m_inv;
        }

        // Step 4: Compute dZ1 = (dZ2 × W2^T) * ReLU_deriv(Z1)
        // First: dZ2 × W2^T → (60000 × 10) × (10 × 128) = (60000 × 128)
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans, // Transpose W2
            @intCast(m), // M = 60000
            @intCast(n_hidden), // N = 128
            @intCast(n_classes), // K = 10
            1.0,
            dZ2.ptr, // (60000 × 10)
            @intCast(n_classes), // lda = 10
            W2.ptr, // (128 × 10), used transposed
            @intCast(n_classes), // ldb = 10 (cols in stored W2)
            0.0,
            dZ1.ptr, // (60000 × 128)
            @intCast(n_hidden), // ldc = 128
        );

        // Then: Element-wise multiply by ReLU derivative
        // ReLU_deriv(z) = 1 if z > 0, else 0
        for (0..m * n_hidden) |i| {
            if (Z1[i] <= 0.0) {
                dZ1[i] = 0.0;
            }
        }

        // Step 5: Compute dW1 = (1/m) * X_train^T × dZ1
        // Result: (784 × 128)
        cblas_dgemm(
            CblasRowMajor,
            CblasTrans, // Transpose X_train
            CblasNoTrans,
            @intCast(k), // M = 784
            @intCast(n_hidden), // N = 128
            @intCast(m), // K = 60000
            m_inv, // alpha = 1/m
            X_train.ptr, // (60000 × 784), used transposed
            @intCast(k), // lda = 784
            dZ1.ptr, // (60000 × 128)
            @intCast(n_hidden), // ldb = 128
            0.0,
            dW1.ptr, // (784 × 128)
            @intCast(n_hidden), // ldc = 128
        );

        // Step 6: Compute db1 = (1/m) * sum(dZ1)
        // Sum all elements
        //var db1_sum: f64 = 0.0;
        for (0..n_hidden) |j| {
            var col_sum: f64 = 0.0;
            for (0..m) |i| {
                col_sum += dZ1[i * n_hidden + j];
            }
            db1[j] = col_sum * m_inv;
        }

        // ============================================
        // PARAMETER UPDATES
        // ============================================

        // Update W1, W2, b1, b2
        for (0..k * n_hidden) |i| {
            W1[i] -= alpha * dW1[i];
        }
        for (0..n_hidden * n_classes) |i| {
            W2[i] -= alpha * dW2[i];
        }
        for (0..n_hidden) |i| {
            b1[i] -= alpha * db1[i];
        }
        for (0..n_classes) |i| {
            b2[i] -= alpha * db2[i];
        }
        get_predictions(A2, predictions, m, n_classes);
        const accuracy = get_accuracy(predictions, Y_train, m);
        try stdout.print("Epoch {d}: Accuracy = {d:.2}%\n", .{ epoch, accuracy });
    }
    const end = std.time.nanoTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;

    try stdout.print("Training completed in {d:.2}ms\n", .{elapsed_ms});

    // Calculate total operations: (forward + backward) × num_epochs
    const forward_ops = 2.0 * @as(f64, @floatFromInt(m * n_hidden * k)) +
        2.0 * @as(f64, @floatFromInt(m * n_classes * n_hidden));
    const backward_ops = 2.0 * @as(f64, @floatFromInt(n_hidden * n_classes * m)) +
        2.0 * @as(f64, @floatFromInt(m * n_hidden * n_classes)) +
        2.0 * @as(f64, @floatFromInt(k * n_hidden * m));
    const total_ops = (forward_ops + backward_ops) * @as(f64, @floatFromInt(num_epochs));

    const gflops = (total_ops / 1e9) / (elapsed_ms / 1000.0);
    try stdout.print("\nPerformance: {d:.2} GFLOPS\n", .{gflops});
}
