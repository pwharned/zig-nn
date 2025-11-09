# Neural Network in Zig

A simple zig learning project  - a neural network implemented in zig / openblas. 

A two-layer neural network implementation in Zig for MNIST digit classification using OpenBLAS for matrix multiplication.

## Features

- Two-layer feedforward neural network (784 -> hidden -> 10)
- ReLU activation for hidden layer
- Softmax activation for output layer
- Hand written backpropagation with gradient descent
```
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
// Then apply: ReLU(Z1) → A1 ( 60000 × 128)
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
// Then apply: Softmax(Z2) → Predictions ( 60000 × 10)
```



## Requirements

- Zig 0.14.0 or later
- OpenBLAS library
- MNIST dataset in CSV format

## Building

`zig build`

## Run

Will accept the number of epochs and the alpha. Typically converges around 90~95% accuracy after 100 epochs with the alpha at 0.9.

`./nn mnist_train.csv 100 0.9`

# Comments

Overall zig is somewhat easier and more intuitive to work with than similar projects in C/C++. After building for release, the speed of initializing the matrices and reading the dataset is FAST. There are more helpful messages when you attempt out of bounds array access or forget to free memory.I plan on attempting to implement a Goto based Gemm algorithm to compliment this.
```
❯ ./zig-out/bin/nn data/data.csv 1000 0.9
Training rate: 9e-1
OpenBLAS config: OpenBLAS 0.3.26 DYNAMIC_ARCH NO_AFFINITY USE_OPENMP SkylakeX MAX_THREADS=128
OpenBLAS threads: 16

Initializing matrices...
Computing C = A * B...
Epoch 1: Accuracy = 9.35%
Epoch 2: Accuracy = 33.55%
Epoch 3: Accuracy = 20.16%
Epoch 4: Accuracy = 32.66%
Epoch 5: Accuracy = 37.28%
Epoch 6: Accuracy = 43.47%
Epoch 7: Accuracy = 46.61%
Epoch 8: Accuracy = 48.88%
Epoch 9: Accuracy = 51.03%
Epoch 10: Accuracy = 52.96%
Epoch 11: Accuracy = 55.45%
Epoch 12: Accuracy = 57.01%
Epoch 13: Accuracy = 58.46%
Epoch 14: Accuracy = 59.81%
Epoch 15: Accuracy = 61.05%
Epoch 16: Accuracy = 62.28%
Epoch 17: Accuracy = 63.25%
Epoch 18: Accuracy = 64.24%
Epoch 19: Accuracy = 65.13%
Epoch 20: Accuracy = 65.91%
Epoch 21: Accuracy = 66.74%
Epoch 22: Accuracy = 67.50%
Epoch 23: Accuracy = 68.16%
Epoch 24: Accuracy = 68.82%
Epoch 25: Accuracy = 69.49%
Epoch 26: Accuracy = 70.13%
Epoch 27: Accuracy = 70.68%
Epoch 28: Accuracy = 71.18%
Epoch 29: Accuracy = 71.65%
Epoch 30: Accuracy = 72.15%
Epoch 31: Accuracy = 72.61%
Epoch 32: Accuracy = 73.03%
Epoch 33: Accuracy = 73.40%
Epoch 34: Accuracy = 73.80%
Epoch 35: Accuracy = 74.19%
Epoch 36: Accuracy = 74.54%
Epoch 37: Accuracy = 74.86%
Epoch 38: Accuracy = 75.18%
Epoch 39: Accuracy = 75.48%
Epoch 40: Accuracy = 75.78%
Epoch 41: Accuracy = 76.10%
Epoch 42: Accuracy = 76.35%
Epoch 43: Accuracy = 76.59%
Epoch 44: Accuracy = 76.84%
Epoch 45: Accuracy = 77.09%
Epoch 46: Accuracy = 77.40%
Epoch 47: Accuracy = 77.63%
Epoch 48: Accuracy = 77.87%
Epoch 49: Accuracy = 78.12%
Epoch 50: Accuracy = 78.32%
Epoch 51: Accuracy = 78.56%
Epoch 52: Accuracy = 78.77%
Epoch 53: Accuracy = 78.97%
Epoch 54: Accuracy = 79.18%
Epoch 55: Accuracy = 79.38%
Epoch 56: Accuracy = 79.60%
Epoch 57: Accuracy = 79.80%
Epoch 58: Accuracy = 79.99%
Epoch 59: Accuracy = 80.16%
Epoch 60: Accuracy = 80.34%
Epoch 61: Accuracy = 80.50%
Epoch 62: Accuracy = 80.69%
Epoch 63: Accuracy = 80.84%
Epoch 64: Accuracy = 81.01%
Epoch 65: Accuracy = 81.15%
Epoch 66: Accuracy = 81.29%
Epoch 67: Accuracy = 81.43%
Epoch 68: Accuracy = 81.58%
Epoch 69: Accuracy = 81.72%
Epoch 70: Accuracy = 81.84%
Epoch 71: Accuracy = 81.97%
Epoch 72: Accuracy = 82.12%
Epoch 73: Accuracy = 82.23%
Epoch 74: Accuracy = 82.33%
Epoch 75: Accuracy = 82.46%
Epoch 76: Accuracy = 82.59%
Epoch 77: Accuracy = 82.72%
Epoch 78: Accuracy = 82.83%
Epoch 79: Accuracy = 82.92%
Epoch 80: Accuracy = 83.00%
Epoch 81: Accuracy = 83.10%
Epoch 82: Accuracy = 83.19%
Epoch 83: Accuracy = 83.30%
Epoch 84: Accuracy = 83.40%
Epoch 85: Accuracy = 83.50%
Epoch 86: Accuracy = 83.60%
Epoch 87: Accuracy = 83.67%
Epoch 88: Accuracy = 83.77%
Epoch 89: Accuracy = 83.87%
Epoch 90: Accuracy = 83.95%
Epoch 91: Accuracy = 84.03%
Epoch 92: Accuracy = 84.12%
Epoch 93: Accuracy = 84.21%
Epoch 94: Accuracy = 84.27%
Epoch 95: Accuracy = 84.36%
Epoch 96: Accuracy = 84.45%
Epoch 97: Accuracy = 84.53%
Epoch 98: Accuracy = 84.59%
Epoch 99: Accuracy = 84.66%
Epoch 100: Accuracy = 84.75%
Epoch 101: Accuracy = 84.81%
Epoch 102: Accuracy = 84.90%
Epoch 103: Accuracy = 84.96%
Epoch 104: Accuracy = 85.01%
Epoch 105: Accuracy = 85.08%
Epoch 106: Accuracy = 85.14%
Epoch 107: Accuracy = 85.20%
Epoch 108: Accuracy = 85.25%
Epoch 109: Accuracy = 85.32%
Epoch 110: Accuracy = 85.37%
Epoch 111: Accuracy = 85.45%
```
