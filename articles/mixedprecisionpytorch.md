# Mixed Precision Computation in PyTorch

## Precision Modes and Configurations

Mixed precision computation leverages different floating-point formats to optimize computational performance and memory efficiency. PyTorch provides several mechanisms to implement mixed precision training and inference:

### Key Precision Formats

1. **FP16 (Half Precision)**
   - 16-bit floating-point format
   - Reduced memory footprint
   - Potential performance gains on modern GPUs
   - Reduced numerical precision

2. **BF16 (Brain Floating Point)**
   - 16-bit format with extended dynamic range
   - Better numerical stability compared to FP16
   - Supported on newer hardware (e.g., NVIDIA Ampere, AMD RDNA)

3. **TF32 (Tensor Float-32)**
   - 32-bit format with reduced precision
   - Native to NVIDIA Ampere architecture
   - Maintains most of FP32's dynamic range

## Implementing Mixed Precision with `torch.cuda.amp`

### Basic Mixed Precision Training Setup

```python
import torch
import torch.nn as nn
import torch.cuda.amp as amp

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.layer(x)

# Create model and optimizer
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

# Gradient scaler for preventing underflow
scaler = amp.GradScaler()

# Training loop with automatic mixed precision
for input_batch in dataloader:
    optimizer.zero_grad()
    
    # Automatic mixed precision context
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(input_batch)
        loss = criterion(output, target)
    
    # Scale loss to prevent underflow
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Precision Configuration Methods

### Global Precision Setting

```python
# Set global precision for CUDA operations
torch.set_float32_matmul_precision('high')  # Options: 'high', 'medium', 'low'
```

### Tensor-Level Precision Conversion

```python
# Convert tensor precision
x_fp16 = x.half()  # Convert to FP16
x_bf16 = x.bfloat16()  # Convert to BF16
x_tf32 = x.to(torch.float32)  # TF32 via standard float
```

## Performance Considerations

- **Memory Efficiency**: Mixed precision can reduce memory usage by 50%
- **Computational Speed**: Up to 2-3x faster on supported hardware
- **Numerical Stability**: Use `GradScaler` to prevent underflow

## Hardware Compatibility

- **NVIDIA GPUs**: 
  - Ampere (A100, RTX 30 series): Native TF32 support
  - Turing, Volta: Partial mixed-precision capabilities
- **AMD GPUs**: Increasing support for mixed precision
- **CPU**: Limited mixed precision support

## Debugging and Monitoring

```python
# Check current precision settings
print(torch.get_float32_matmul_precision())

# Validate numerical stability
torch.autograd.set_detect_anomaly(True)
```

## Best Practices

1. Start with `torch.cuda.amp.autocast()`
2. Use `GradScaler` for numerical stability
3. Validate model convergence
4. Profile performance before widespread adoption