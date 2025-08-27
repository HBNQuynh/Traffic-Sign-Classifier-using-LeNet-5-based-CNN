import numpy as np
from numba import cuda, float32

# =========================
# Numpy version
# =========================
def relu(x): 
    """Applies the ReLU activation function: max(0, x)."""
    return np.maximum(0, x)
        
def softmax(x):
    """
    Applies the softmax activation function.
    Args:
        x (ndarray): Input array of shape (batch, num_classes).
    Returns:
        ndarray: Probabilities of shape (batch, num_classes).
    """
    x = x - np.max(x, axis=1, keepdims=True)  # for numerical stability
    exp_vals = np.exp(x)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def max_pool2d_numpy_vec(x, size=2, stride=2):
    """
    Performs 2D max pooling on a 4D tensor.
    Args:
        x (ndarray): Input tensor (B, C, H, W).
        size (int): Pooling window size.
        stride (int): Step size.
    Returns:
        ndarray: Pooled tensor.
    """
    B, C, H, W = x.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (size, size), axis=(2, 3))
    strided = windows[:, :, ::stride, ::stride, :, :]
    return strided.reshape(B, C, out_h, out_w, -1).max(axis=-1)

def im2col(x, kH, kW, stride, padding):
    """
    Converts input tensor into columns for efficient convolution.
    Args:
        x (ndarray): Input tensor (B, C_in, H, W).
        kH, kW (int): Kernel size.
        stride (int): Stride.
        padding (int): Zero-padding.
    Returns:
        cols (ndarray): Transformed columns.
        H_out, W_out (int): Output height and width.
    """
    B, C_in, H, W = x.shape
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), 'constant')
    windows = np.lib.stride_tricks.sliding_window_view(x, (kH, kW), axis=(2,3))
    windows = windows[:, :, ::stride, ::stride, :, :]
    cols = windows.transpose(0,2,3,1,4,5).reshape(B * H_out * W_out, C_in * kH * kW)
    return cols, H_out, W_out

def conv2d_numpy_vec(x, weight, bias, stride=1, padding=0):
    """
    Performs 2D convolution using im2col.
    Args:
        x (ndarray): Input tensor (B, C_in, H, W).
        weight (ndarray): Convolution kernels (C_out, C_in, kH, kW).
        bias (ndarray): Bias vector (C_out,).
    Returns:
        ndarray: Output tensor (B, C_out, H_out, W_out).
    """
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    cols, H_out, W_out = im2col(x, kH, kW, stride, padding)
    out = cols @ weight.reshape(C_out, -1).T + bias
    return out.reshape(B, H_out, W_out, C_out).transpose(0,3,1,2)


class SimpleCNN_Sequential_Vec:
    """
    A simple CNN model implemented in NumPy.
    Architecture:
        Conv1 -> ReLU -> MaxPool
        Conv2 -> ReLU -> MaxPool
        Flatten -> FC1 -> ReLU -> FC2 -> Softmax
    """
    def __init__(self, input_shape=(3,32,32), num_classes=43):
        rng = np.random.default_rng()
        flatten_size = 32 * 8 * 8

        # Xavier/He initialization for weights
        self.params = {
            "conv1_w": (rng.standard_normal((16,3,3,3)) * np.sqrt(2/(3*3*3))).astype(np.float32),
            "conv1_b": np.zeros(16, np.float32),
            "conv2_w": (rng.standard_normal((32,16,3,3)) * np.sqrt(2/(16*3*3))).astype(np.float32),
            "conv2_b": np.zeros(32, np.float32),
            "fc1_w": (rng.standard_normal((flatten_size,128)) * np.sqrt(2/flatten_size)).astype(np.float32),
            "fc1_b": np.zeros(128, np.float32),
            "fc2_w": (rng.standard_normal((128,num_classes)) * np.sqrt(2/128)).astype(np.float32),
            "fc2_b": np.zeros(num_classes, np.float32),
        }

    def forward(self, x):
        """Forward pass through the CNN."""
        c1 = conv2d_numpy_vec(x, self.params["conv1_w"], self.params["conv1_b"], 1, 1)
        r1 = relu(c1)
        p1 = max_pool2d_numpy_vec(r1)
        c2 = conv2d_numpy_vec(p1, self.params["conv2_w"], self.params["conv2_b"], 1, 1)
        r2 = relu(c2)
        p2 = max_pool2d_numpy_vec(r2)
        flat = p2.reshape(p2.shape[0], -1)
        fc1 = relu(flat @ self.params["fc1_w"] + self.params["fc1_b"])
        out = softmax(fc1 @ self.params["fc2_w"] + self.params["fc2_b"])
        return out

    def predict(self, x):
        """
        Runs inference and returns predicted class indices.
        Args:
            x (ndarray): Input images (N,H,W,C) or (N,C,H,W).
        Returns:
            ndarray: Predicted class indices (N,).
        """
        if x.ndim == 4 and x.shape[-1] == 3:  # (N,H,W,C) -> (N,C,H,W)
            x = np.transpose(x, (0,3,1,2))
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
    

# =========================
# Final version
# =========================
@cuda.jit
def pad2d_cuda_kernel(x, out, p):
    """
    Pads a 4D tensor `x` with `p` pixels of zero-padding on height and width.
    The result is stored in `out`.
    """
    idx = cuda.grid(1)

    if idx >= out.size:
        return

    j = idx % out.shape[3]
    i = (idx // out.shape[3]) % out.shape[2]
    c = (idx // (out.shape[3] * out.shape[2])) % out.shape[1]
    b = idx // (out.shape[3] * out.shape[2] * out.shape[1])
    i_in, j_in = i - p, j - p

    if (0 <= i_in < x.shape[2]) and (0 <= j_in < x.shape[3]):
        out[b, c, i, j] = x[b, c, i_in, j_in]
    else:
        out[b, c, i, j] = 0.0

@cuda.jit
def conv2d_relu_shared_mem(x_padded, weight, bias, out, stride):
    """
    Fused 2D Convolution + Bias + ReLU using shared memory with 2D thread indexing.
    - Threads per block: (16, 16)
    - Grid dims: (ceil(W_out/16), ceil(H_out/16), B * C_out)
      * blockIdx.z encodes (batch, out_channel) as a flattened index.
    """
    s_x = cuda.shared.array(shape=(18, 18), dtype=float32)  # (TILE + K - 1) with TILE=16, K=3
    s_w = cuda.shared.array(shape=(3, 3), dtype=float32)

    # 2D global indices for output spatial coords
    j_out, i_out = cuda.grid(2)  # (x, y) -> (col, row)

    # Flattened (batch, out_channel) from grid z-dimension
    b_c_flat = cuda.blockIdx.z
    b = b_c_flat // out.shape[1]      # batch index
    c_out = b_c_flat % out.shape[1]   # output channel index

    if b >= out.shape[0] or c_out >= out.shape[1]:
        return

    # Top-left corner of the output tile computed by this block
    start_row_out = cuda.blockIdx.y * 16
    start_col_out = cuda.blockIdx.x * 16

    # Corresponding top-left corner in the (already padded) input
    start_row_in = start_row_out * stride
    start_col_in = start_col_out * stride

    # Thread-local indices inside the tile
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Flat thread index for cooperative loads
    thread_idx_flat = ty * 16 + tx
    num_threads = 16 * 16

    # Accumulator for this output element
    conv_sum = float32(0.0)

    # Loop over input channels
    for c_in in range(x_padded.shape[1]):
        # 1) Load input tile into shared memory (18x18 covers 16x16 tile + 3x3 halo)
        for t in range(thread_idx_flat, 18 * 18, num_threads):
            s_row = t // 18
            s_col = t % 18
            g_row = start_row_in + s_row
            g_col = start_col_in + s_col
            if g_row < x_padded.shape[2] and g_col < x_padded.shape[3]:
                s_x[s_row, s_col] = x_padded[b, c_in, g_row, g_col]
            else:
                s_x[s_row, s_col] = 0.0

        # 2) Load 3x3 kernel weights for (c_out, c_in) into shared memory
        if thread_idx_flat < 9:
            ki = thread_idx_flat // 3
            kj = thread_idx_flat % 3
            s_w[ki, kj] = weight[c_out, c_in, ki, kj]

        cuda.syncthreads()

        # 3) Convolution using shared memory
        if i_out < out.shape[2] and j_out < out.shape[3]:
            # ty/tx are within [0,15], so ty+ki, tx+kj are safe within 18x18
            for ki in range(3):
                for kj in range(3):
                    conv_sum += s_x[ty + ki, tx + kj] * s_w[ki, kj]

        cuda.syncthreads()

    # 4) Bias + ReLU + write-back
    if i_out < out.shape[2] and j_out < out.shape[3]:
        val = conv_sum + bias[c_out]
        out[b, c_out, i_out, j_out] = val if val > 0.0 else 0.0

@cuda.jit
def max_pool2d_cuda_kernel(x, out, size, stride):
    """
    Performs 2D max pooling over the input tensor `x`.
    """
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    b_c_flat = cuda.blockIdx.z
    b = b_c_flat // out.shape[1]
    c = b_c_flat % out.shape[1]

    if b < out.shape[0] and c < out.shape[1] and i < out.shape[2] and j < out.shape[3]:
        max_val = -float32(1e9)
        for ki in range(size):
            for kj in range(size):
                h_idx = i * stride + ki
                w_idx = j * stride + kj
                if h_idx < x.shape[2] and w_idx < x.shape[3]:
                    val = x[b, c, h_idx, w_idx]
                    if val > max_val:
                        max_val = val
        out[b, c, i, j] = max_val

@cuda.jit
def matmul_forward_shared_mem(A, B, C):
    """
    Performs matrix multiplication C = A @ B using shared memory tiling.
    """
    # The shape for cuda.shared.array MUST be a literal constant tuple.
    sA = cuda.shared.array(shape=(16, 17), dtype=float32) # Bank conflict resolved
    sB = cuda.shared.array(shape=(16, 17), dtype=float32)

    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y

    row = by * 16 + ty
    col = bx * 16 + tx
    acc = float32(0.0)

    # Loop over the tiles of A and B required to compute the C tile
    for i in range((A.shape[1] + 15) // 16):
        # Load tiles from global to shared memory
        if row < A.shape[0] and (i * 16 + tx) < A.shape[1]:
            sA[ty, tx] = A[row, i * 16 + tx]
        else:
            sA[ty, tx] = 0.0

        if col < B.shape[1] and (i * 16 + ty) < B.shape[0]:
            sB[ty, tx] = B[i * 16 + ty, col]
        else:
            sB[ty, tx] = 0.0

        cuda.syncthreads() # Wait for all threads to finish loading

        # Multiply the two tiles from shared memory
        for k in range(16):
            acc += sA[ty, k] * sB[k, tx]
        cuda.syncthreads() # Wait for all threads to finish computing

    # Write the final result to global memory
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = acc

@cuda.jit
def matmul_add_bias_cuda_1D(A, b, out):
    """
    Adds a bias vector `b` to each row of matrix `A` and stores in `out`.
    """
    idx = cuda.grid(1)

    if idx >= out.size:
        return

    j = idx % out.shape[1]
    i = idx // out.shape[1]
    out[i, j] = A[i, j] + b[j]

@cuda.jit
def relu_cuda(x, out):
    i = cuda.grid(1)
    if i < x.size:
        out.flat[i] = max(0.0, x.flat[i])


class SimpleCNN_Final():
    def __init__(self, input_shape=(3, 32, 32), num_classes=43):
        """
        Initialize a simple CNN model with 2 convolutional layers and 2 FC layers.
        All parameters are initialized on CPU then transferred to GPU.
        """
        self.num_classes = num_classes
        self.params_cpu = {}
        self.params_gpu = {}
        self.gradients_gpu = {}

        # Layer 1 (Conv1): 3 input channels -> 16 filters of size 3x3
        self.params_cpu['conv1_w'] = np.random.randn(16, 3, 3, 3).astype(np.float32) * np.sqrt(2.0 / (3 * 3 * 3))
        self.params_cpu['conv1_b'] = np.zeros(16, dtype=np.float32)

        # Layer 2 (Conv2): 16 input channels -> 32 filters of size 3x3
        self.params_cpu['conv2_w'] = np.random.randn(32, 16, 3, 3).astype(np.float32) * np.sqrt(2.0 / (16 * 3 * 3))
        self.params_cpu['conv2_b'] = np.zeros(32, dtype=np.float32)

        # FC layers
        flatten_size = 32 * 8 * 8
        self.params_cpu['fc1_w'] = np.random.randn(flatten_size, 128).astype(np.float32) * np.sqrt(2.0 / flatten_size)
        self.params_cpu['fc1_b'] = np.zeros(128, dtype=np.float32)
        self.params_cpu['fc2_w'] = np.random.randn(128, num_classes).astype(np.float32) * np.sqrt(2.0 / 128)
        self.params_cpu['fc2_b'] = np.zeros(num_classes, dtype=np.float32)

        # Transfer weights to GPU
        for name, param in self.params_cpu.items():
            self.params_gpu[name] = cuda.to_device(param)
            # Initialize gradients on GPU
            self.gradients_gpu['grad_' + name] = cuda.device_array_like(self.params_gpu[name])

        self.intermediate_values_gpu = {}
        self.layer_timings = {}

    def _log_and_time_kernel(self, kernel_name, kernel, blockspergrid, threadsperblock, *args):
        """
        Utility to measure execution time of a CUDA kernel.
        Stores time in `self.layer_timings`.
        """
        kernel[blockspergrid, threadsperblock](*args)


    def forward(self, x_cpu):
        """
        Forward pass using fused Conv+Bias+ReLU kernels for conv layers.
        """
        self.layer_timings = {}
        self.intermediate_values_gpu = {}

        x_gpu = cuda.to_device(x_cpu)
        self.intermediate_values_gpu['x'] = x_gpu
        batch_size = x_cpu.shape[0]

        TPB_2D_SHARED = (16, 16)
        THREADS_1D = 256

        # --- Layer 1: Conv1 (Fused Conv+ReLU) + MaxPool ---
        conv1_padded = cuda.device_array((batch_size, 3, 34, 34), np.float32)
        grid_1d_pad1 = (conv1_padded.size + THREADS_1D - 1) // THREADS_1D
        self._log_and_time_kernel('pad1_forward', pad2d_cuda_kernel, grid_1d_pad1, THREADS_1D, x_gpu, conv1_padded, 1)
        self.intermediate_values_gpu['conv1_input_padded'] = conv1_padded

        relu1_out = cuda.device_array((batch_size, 16, 32, 32), np.float32)
        grid_conv1 = ((32 + 15) // 16, (32 + 15) // 16, batch_size * 16)
        self._log_and_time_kernel('conv1_fused_forward', conv2d_relu_shared_mem, grid_conv1, TPB_2D_SHARED,
                          conv1_padded, self.params_gpu['conv1_w'], self.params_gpu['conv1_b'], relu1_out, 1)
        self.intermediate_values_gpu['conv1_out'] = relu1_out
        self.intermediate_values_gpu['relu1_out'] = relu1_out  # kept for compatibility

        pool1_out = cuda.device_array((batch_size, 16, 16, 16), np.float32)
        grid_pool1 = ((16 + 15) // 16, (16 + 15) // 16, batch_size * 16)
        self._log_and_time_kernel('pool1_forward', max_pool2d_cuda_kernel, grid_pool1, TPB_2D_SHARED,
                          relu1_out, pool1_out, 2, 2)
        self.intermediate_values_gpu['pool1_out'] = pool1_out

        # --- Layer 2: Conv2 (Fused Conv+ReLU) + MaxPool ---
        conv2_padded = cuda.device_array((batch_size, 16, 18, 18), np.float32)
        grid_1d_pad2 = (conv2_padded.size + THREADS_1D - 1) // THREADS_1D
        self._log_and_time_kernel('pad2_forward', pad2d_cuda_kernel, grid_1d_pad2, THREADS_1D, pool1_out, conv2_padded, 1)
        self.intermediate_values_gpu['conv2_input_padded'] = conv2_padded

        relu2_out = cuda.device_array((batch_size, 32, 16, 16), np.float32)
        grid_conv2 = ((16 + 15) // 16, (16 + 15) // 16, batch_size * 32)
        self._log_and_time_kernel('conv2_fused_forward', conv2d_relu_shared_mem, grid_conv2, TPB_2D_SHARED,
                          conv2_padded, self.params_gpu['conv2_w'], self.params_gpu['conv2_b'], relu2_out, 1)
        self.intermediate_values_gpu['conv2_out'] = relu2_out
        self.intermediate_values_gpu['relu2_out'] = relu2_out

        pool2_out = cuda.device_array((batch_size, 32, 8, 8), np.float32)
        grid_pool2 = ((8 + 15) // 16, (8 + 15) // 16, batch_size * 32)
        self._log_and_time_kernel('pool2_forward', max_pool2d_cuda_kernel, grid_pool2, TPB_2D_SHARED,
                          relu2_out, pool2_out, 2, 2)
        self.intermediate_values_gpu['pool2_out'] = pool2_out

        # --- FC Layers ---
        flatten_out = pool2_out.reshape(batch_size, -1)
        self.intermediate_values_gpu['flatten_out'] = flatten_out

        fc1_linear = cuda.device_array((batch_size, 128), np.float32)
        grid_fc1 = ((128 + 15) // 16, (batch_size + 15) // 16)
        self._log_and_time_kernel('fc1_matmul_forward_shared', matmul_forward_shared_mem, grid_fc1, TPB_2D_SHARED,
                          flatten_out, self.params_gpu['fc1_w'], fc1_linear)

        fc1_with_bias = cuda.device_array_like(fc1_linear)
        grid_1d_fc1 = (fc1_linear.size + THREADS_1D - 1) // THREADS_1D
        self._log_and_time_kernel('fc1_bias_add_forward', matmul_add_bias_cuda_1D, grid_1d_fc1, THREADS_1D,
                          fc1_linear, self.params_gpu['fc1_b'], fc1_with_bias)
        self.intermediate_values_gpu['fc1_linear'] = fc1_with_bias

        fc1_out = cuda.device_array_like(fc1_with_bias)
        grid_relu3 = (fc1_out.size + THREADS_1D - 1) // THREADS_1D
        self._log_and_time_kernel('relu3_forward', relu_cuda, grid_relu3, THREADS_1D,
                          fc1_with_bias, fc1_out)
        self.intermediate_values_gpu['fc1_out'] = fc1_out

        fc2_linear = cuda.device_array((batch_size, self.num_classes), np.float32)
        grid_fc2 = ((self.num_classes + 15) // 16, (batch_size + 15) // 16)
        self._log_and_time_kernel('fc2_matmul_forward_shared', matmul_forward_shared_mem, grid_fc2, TPB_2D_SHARED,
                          fc1_out, self.params_gpu['fc2_w'], fc2_linear)

        fc2_with_bias = cuda.device_array_like(fc2_linear)
        grid_1d_fc2 = (fc2_linear.size + THREADS_1D - 1) // THREADS_1D
        self._log_and_time_kernel('fc2_bias_add_forward', matmul_add_bias_cuda_1D, grid_1d_fc2, THREADS_1D,
                          fc2_linear, self.params_gpu['fc2_b'], fc2_with_bias)
        self.intermediate_values_gpu['fc2_linear'] = fc2_with_bias

        # --- Softmax on CPU ---
        logits = fc2_with_bias.copy_to_host()
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)

        return probs


    def predict(self, x):
        """
        Predicts the class label from input x using forward pass.
        """
        probs = self.forward(x)
        return np.argmax(probs, axis=1)
