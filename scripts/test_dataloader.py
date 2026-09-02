import os
import sys
import time
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from dataset import create_dataloader

def test_dataloader():
    print("==================================================")
    print("PYTORCH DATASET & DATALOADER VERIFICATION")
    print("==================================================\n")

    train_manifest = os.path.join(project_root, "data", "splits", "train.txt")
    val_manifest = os.path.join(project_root, "data", "splits", "val.txt")
    test_manifest = os.path.join(project_root, "data", "splits", "test.txt")

    batch_size = 8
    num_workers = 0
    pin_memory = torch.cuda.is_available()
    
    # 1. Create dataloaders
    train_loader = create_dataloader(train_manifest, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = create_dataloader(val_manifest, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = create_dataloader(test_manifest, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    test_len = len(test_loader.dataset)

    print(f"Train Dataset: {train_len}")
    print(f"Validation Dataset: {val_len}")
    print(f"Test Dataset: {test_len}\n")

    if train_len != 10000 or val_len != 1000 or test_len != 500:
        print("ERROR: Dataset lengths do not match expected values (10000, 1000, 500)!")
        sys.exit(1)

    # 2. Verify one batch from each
    loaders = [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]
    batch_shape_pass = True
    finite_test_pass = True

    for name, loader in loaders:
        print(f"--- {name} Batch ---")
        batch_L, batch_ab = next(iter(loader))
        
        print(f"L shape: {list(batch_L.shape)}")
        print(f"ab shape: {list(batch_ab.shape)}")
        print(f"L dtype: {batch_L.dtype}")
        print(f"ab dtype: {batch_ab.dtype}")
        print(f"L device: {batch_L.device}")
        print(f"ab device: {batch_ab.device}")
        
        if list(batch_L.shape) != [batch_size, 1, 256, 256] or list(batch_ab.shape) != [batch_size, 2, 256, 256]:
            batch_shape_pass = False
        
        if batch_L.dtype != torch.float32 or batch_ab.dtype != torch.float32:
            batch_shape_pass = False

        if not (torch.isfinite(batch_L).all() and torch.isfinite(batch_ab).all()):
            finite_test_pass = False
            
        print(f"L min/max: {batch_L.min().item():.4f} / {batch_L.max().item():.4f}")
        print(f"ab min/max: {batch_ab.min().item():.4f} / {batch_ab.max().item():.4f}\n")

    print(f"Batch Shape Test: {'PASS' if batch_shape_pass else 'FAIL'}")
    print(f"Finite Value Test: {'PASS' if finite_test_pass else 'FAIL'}\n")

    if not batch_shape_pass or not finite_test_pass:
        sys.exit(1)

    # 3. GPU/Accelerator Transfer Test
    from device import get_device, get_device_name, empty_device_cache, get_memory_stats
    device = get_device()
    print(f"Accelerator Device: {device} ({get_device_name(device)})")
    
    if device.type in ('cuda', 'mps'):
        empty_device_cache(device)
        batch_L, batch_ab = next(iter(train_loader))
        
        batch_L_acc = batch_L.to(device)
        batch_ab_acc = batch_ab.to(device)
        
        acc_transfer_pass = (batch_L_acc.device.type == device.type and batch_ab_acc.device.type == device.type)
        print(f"Accelerator Batch Transfer Test: {'PASS' if acc_transfer_pass else 'FAIL'}")
        
        mem_stats = get_memory_stats(device)
        if 'allocated_mb' in mem_stats:
            print(f"Memory Allocated: {mem_stats['allocated_mb']:.2f} MB")
        if 'reserved_mb' in mem_stats:
            print(f"Memory Reserved: {mem_stats['reserved_mb']:.2f} MB")
        elif 'driver_allocated_mb' in mem_stats:
            print(f"Driver Memory Allocated: {mem_stats['driver_allocated_mb']:.2f} MB")
        print()
    else:
        print("Accelerator Batch Transfer Test: SKIPPED (CPU only)\n")

    # 4. Performance Check
    print("Measuring batch loading performance...")
    num_batches = 5
    start_time = time.time()
    
    train_iter = iter(train_loader)
    for _ in range(num_batches):
        _ = next(train_iter)
        
    end_time = time.time()
    avg_time = (end_time - start_time) / num_batches
    print(f"Average batch loading time: {avg_time:.4f} seconds")

if __name__ == "__main__":
    test_dataloader()
