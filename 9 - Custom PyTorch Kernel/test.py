import torch
import my_reduction
import time

print("Correctness Check")
x = torch.ones(1 << 24, device='cuda')

custom = my_reduction.forward(x).item()
reference = x.sum().item()

print(f"Custom kernel : {custom:.0f}")
print(f"torch.sum() :   {reference:.0f}")
print(f"Expected :      {1 << 24}")


print("Benchmark Check")
print(f"{'Size':>6}  {'Custom(ms)':>12}  {'torch.sum(ms)':>13}  {'Ratio':>7}")
print("-" * 44)

for N, name in [(1<<20,'1M'), (1<<22,'4M'), (1<<24,'16M'), (1<<26,'64M')]:
    x = torch.ones(N, device='cuda')

    # Warm up
    for _ in range(10):
        my_reduction.forward(x)
        x.sum()
    torch.cuda.synchronize()

    # Time custom
    t0 = time.perf_counter()
    for _ in range(100):
        my_reduction.forward(x)
    torch.cuda.synchronize()
    custom_ms = (time.perf_counter() - t0) / 100 * 1000

    # Time torch.sum
    t0 = time.perf_counter()
    for _ in range(100):
        x.sum()
    torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - t0) / 100 * 1000

    print(f"{name:>6}  {custom_ms:>12.4f}  {torch_ms:>13.4f}  {custom_ms/torch_ms:>6.2f}x")

print("\nNote : torch.sum() is faster as it uses more optimisation")
print("and years of hardware tuning. This kernel shows the")
print("correct methodology : write kernel, wrap, benchmark.")