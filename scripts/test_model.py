import torch
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from model import ColorizationUNet
from dataset import create_dataloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    print("==================================================")
    print("STEP 4 — U-NET ARCHITECTURE VERIFICATION")
    print("==================================================\n")

    model = ColorizationUNet()
    
    # Dummy Input
    x = torch.randn(1, 1, 256, 256)
    model.eval()
    
    with torch.no_grad():
        out, shapes = model(x, return_shapes=True)
        
    print("ARCHITECTURE")
    print("--------------------------------------------------")
    print(f"Input: {shapes['Input']}")
    print(f"Output: {shapes['Final Output']}")
    print(f"Encoder Channels: 1 -> 64 -> 128 -> 256 -> 512 -> 512 -> 512")
    print(f"Bottleneck: {shapes['Bottleneck']}")
    print("Decoder Channels: Halving sequentially according to skip connections.\n")
    
    print("SPATIAL DIMENSIONS")
    print("--------------------------------------------------")
    for k in ['Input', 'Encoder 1', 'Encoder 2', 'Encoder 3', 'Encoder 4', 'Encoder 5', 'Bottleneck', 
              'Decoder 1', 'Decoder 2', 'Decoder 3', 'Decoder 4', 'Decoder 5', 'Final Output']:
        print(f"{k}: {shapes[k]}")
    print("\n")
    
    print("SKIP CONNECTIONS")
    print("--------------------------------------------------")
    skip_test_pass = True
    for i in range(1, 6):
        s = shapes[f'Skip {i}']
        print(f"Skip {i}:")
        print(f"  Encoder feature shape: {s['enc']}")
        print(f"  Decoder feature shape before concatenation: {s['dec_before']}")
        print(f"  Concatenated shape: {s['concat']}")
        if s['enc'][2:] != s['dec_before'][2:]:
            skip_test_pass = False
    
    print(f"Skip Connection Test: {'PASS' if skip_test_pass else 'FAIL'}\n")

    print("MODEL")
    print("--------------------------------------------------")
    params = count_parameters(model)
    approx_mem_mb = (params * 4) / (1024 ** 2)
    print(f"Trainable Parameters: {params:,}")
    print(f"Approximate FP32 Parameter Memory: {approx_mem_mb:.2f} MB\n")

    print("FORWARD TEST")
    print("--------------------------------------------------")
    print(f"Dummy Input: {list(x.shape)}")
    print(f"Dummy Output: {list(out.shape)}")
    out_shape_pass = (list(out.shape) == [1, 2, 256, 256])
    print(f"Output Shape Test: {'PASS' if out_shape_pass else 'FAIL'}\n")

    print("OUTPUT VALIDATION")
    print("--------------------------------------------------")
    out_min = out.min().item()
    out_max = out.max().item()
    print(f"Output Min: {out_min:.4f}")
    print(f"Output Max: {out_max:.4f}")
    
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(f"NaN: {has_nan}")
    print(f"Inf: {has_inf}")
    
    range_test_pass = (out_min >= -1.01 and out_max <= 1.01 and not has_nan and not has_inf)
    print(f"Output Range Test: {'PASS' if range_test_pass else 'FAIL'}\n")

    print("ACCELERATOR / GPU TEST")
    print("--------------------------------------------------")
    from device import get_device, get_device_name, empty_device_cache, device_synchronize, get_memory_stats
    device = get_device()
    print(f"Device: {device} ({get_device_name(device)})")
    
    gpu_forward_pass = False
    batch_forward_pass = False
    
    if device.type in ('cuda', 'mps'):
        empty_device_cache(device)
        model = model.to(device)
        x_gpu = x.to(device)
        
        try:
            with torch.no_grad():
                out_gpu = model(x_gpu)
            gpu_forward_pass = True
            print("Single Image Forward: PASS")
        except Exception as e:
            print(f"Single Image Forward: FAIL ({e})")
            
        train_manifest = os.path.join(project_root, "data", "splits", "train.txt")
        train_loader = create_dataloader(train_manifest, batch_size=8, shuffle=False)
        
        batch_L, batch_ab = next(iter(train_loader))
        batch_L = batch_L.to(device)
        
        try:
            with torch.no_grad():
                pred = model(batch_L)
            
            if list(pred.shape) == [8, 2, 256, 256]:
                batch_forward_pass = True
                print("Batch 8 Forward: PASS")
            else:
                print(f"Batch 8 Forward: FAIL (Shape was {list(pred.shape)})")
        except Exception as e:
            print(f"Batch 8 Forward: FAIL ({e})")
            
        mem_stats = get_memory_stats(device)
        if 'allocated_mb' in mem_stats:
            print(f"Memory Allocated: {mem_stats['allocated_mb']:.2f} MB")
        if 'reserved_mb' in mem_stats:
            print(f"Memory Reserved: {mem_stats['reserved_mb']:.2f} MB")
        elif 'driver_allocated_mb' in mem_stats:
            print(f"Driver Memory Allocated: {mem_stats['driver_allocated_mb']:.2f} MB")
        print()
        
        print("PERFORMANCE")
        print("--------------------------------------------------")
        # Warmup
        _ = model(batch_L)
        
        device_synchronize(device)
        start_time = time.time()
        num_iters = 5
        for _ in range(num_iters):
            _ = model(batch_L)
        device_synchronize(device)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iters
        print(f"Average Forward Pass: {avg_time:.4f} seconds\n")
    else:
        print("Accelerator Tests Skipped (CPU only)\n")
        
    print("==================================================")
    if skip_test_pass and out_shape_pass and range_test_pass and (gpu_forward_pass and batch_forward_pass if device.type in ('cuda', 'mps') else True):
        print("STEP 4 STATUS: PASS")
    else:
        print("STEP 4 STATUS: FAIL")
    print("==================================================")

if __name__ == "__main__":
    test_model()
