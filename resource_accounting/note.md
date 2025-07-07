- By defalut, tensors are stored in GPU memory
- Move the tensor to GPU memory (device: 0)
- Note that some views are non-contiguous entries.

```python
# Define two tensors
x : Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y : Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)

# Old Way
z = x @ y.transpose(-2, -1)  # z is a view of x and y

# New (einops) Way
from einops import einsum
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

- Dimensions that are not named in the output as summed over.

- you can reduce a single tensor via some operation (e.g., sum, mean, max, min)

x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

y = x.mean(dim=-1)

y = reduce(x, "... hidden -> ...", "mean")

```python
# rearange
from einops import rearrange, einsum, reduce
from einops.typing import Float
import torch
x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)

w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads = 2)

x = einsum(x, w, "... hidden1, hidden1, hidden2 -> ... hidden2")

x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

# Computational Cost
FLOPs: floating-point operations (measure of compputation done)
FLOP/s: number of floating-point operations per second (measure of performance)

- matrix multiplication: actual_num_flops = 2 * m * n * k

- Model FLOPs utilization (MFU): ratio of the number of floating-point operations performed by the model to the number of floating-point operations that could be performed in a given time period. 

Usually, MFU of >= 0.5 is quite good.

# Summary
- Matrix multiplication dominate: (2 m n p) FLOPs
- FLOP/s depends on hardware (H100 >> A100) and data type (bfloat16 >> float32)
- Model FLOPs utilization (MFU): (actual FLOP/s) / (promised FLOP/s)

# gradients
- Forward pass: compute loss
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)

loss.backward()

assert w.grad is None


# Gradient accumulation

Recall model: x--w1 --> h1--w2 --> h2 -> loss

- h1.grad = d loss / dh1
- h2.grad = d loss / dh2
- w1.grad = d loss / dw1
- w2.grad = d loss / dw2

Focus on w2 (D*K matrix):
h2.grad[i, k] = d loss / dh2[i, k]

w2.grad[j, k] = sum_i h1[i, j] * h2.grad[i, k]

对于 $w2.grad$ 的计算，涉及到的 FLOPs 数量为 $2 \times m \times n \times p$，其中 $m$ 是 batch size，$n$ 是 h1 的维度，$p$ 是 h2 的维度。每个 $w2[j, k]$ 需要对 $i$ 累加（一次乘法和一次加法），总共 $n \times p$ 个 $w2$ 元素，每个元素累加 $m$ 次，因此总 FLOPs 为 $2 \times m \times n \times p$。

h1.grad[i, j] = sum_k h2.grad[i, k] * w2[j, k]


w1.grad = d loss / dw1

w1.grad[j, k] = sum_i h1.grad[i, j] * x[i, k]

w1 (D*D Parameter matrix):


Putting it together:
- Forward pass: 2 (# data points) (# parameters) FLOPs
- Backward pass: 4 (# data points) (# parameters) FLOPs
- Total: 6 (# data points) (# parameters) FLOPs

# Models

x = nn.Parameter(torch.randn(input_dim))

output = x @ w

w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

Xavier initialization: w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

To be extra safe, we truncate the normal distribution to [-3, 3].
w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), mean=0, std=1, a=-3, b=3) / np.sqrt(input_dim))


data = np.memmap("data.npy", dtype=np.int32)

B = 2
L = 4
x = get_batch(data, batch_size=B, seq_len=L, device=get_device())

AdaGrad optimizer:

- momentum: v = beta * v + (1 - beta) * grad
- AdaGrad: v = v + grad^2
- RMSProp: v = beta * v + (1 - beta) * grad^2
- Adam: m = beta1 * m + (1 - beta1) * grad


A concrete plan:
- Use {bfloat16, fp8} for the forward pass (activations)
- Use float32 for the rest (parameters, gradients)