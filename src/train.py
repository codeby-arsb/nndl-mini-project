import os
import sys
import time
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "src"))

from model import ColorizationUNet
from dataset import create_dataloader

def get_args():
    parser = argparse.ArgumentParser(description="Train Colorization U-Net")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short smoke test instead of full training")
    parser.add_argument("--amp", action="store_true", help="Use Automatic Mixed Precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and torch.cuda.is_available()
    
    print("==================================================")
    print("TRAINING CONFIGURATION")
    print("==================================================")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {'1 (Smoke Test)' if args.smoke_test else args.epochs}")
    print(f"AMP: {use_amp}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("==================================================\n")
    
    # Data loaders
    train_manifest = os.path.join(project_root, "data", "splits", "train.txt")
    val_manifest = os.path.join(project_root, "data", "splits", "val.txt")
    
    train_loader = create_dataloader(train_manifest, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = create_dataloader(val_manifest, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model
    model = ColorizationUNet().to(device)
    
    # Loss, Optimizer, Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume Checkpoint
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch} with best_val_loss {best_val_loss:.4f}\n")
    
    # Output paths
    checkpoints_dir = os.path.join(project_root, "outputs", "checkpoints")
    plots_dir = os.path.join(project_root, "outputs", "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    history_file = os.path.join(project_root, "outputs", "training_history.csv")
    write_header = not os.path.exists(history_file) or start_epoch == 0
    
    history = {'train_loss': [], 'val_loss': []}
    
    # Smoke Test Restrictions
    num_train_batches = 5 if args.smoke_test else len(train_loader)
    num_val_batches = 2 if args.smoke_test else len(val_loader)
    num_epochs = start_epoch + 1 if args.smoke_test else args.epochs
    
    # For smoke test parameter check
    test_param = next(model.parameters())
    pre_step_param = None
    param_updated = False

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = 0.0
        
        # Training Loop
        for i, (l_batch, ab_batch) in enumerate(train_loader):
            if i >= num_train_batches:
                break
                
            l_batch, ab_batch = l_batch.to(device), ab_batch.to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred_ab = model(l_batch)
                    loss = criterion(pred_ab, ab_batch)
            else:
                pred_ab = model(l_batch)
                loss = criterion(pred_ab, ab_batch)
                
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[ERROR] Non-finite loss detected at Epoch {epoch+1}, Batch {i+1}: {loss.item()}")
                sys.exit(1)
                
            if use_amp:
                scaler.scale(loss).backward()
                if args.smoke_test and i == 0:
                    pre_step_param = test_param.clone().detach()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.smoke_test and i == 0:
                    pre_step_param = test_param.clone().detach()
                optimizer.step()
                
            if args.smoke_test and i == 0:
                if not torch.equal(test_param, pre_step_param):
                    param_updated = True
                
            running_loss += loss.item()
            
        train_loss = running_loss / num_train_batches
        
        # Validation Loop
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for i, (l_batch, ab_batch) in enumerate(val_loader):
                if i >= num_val_batches:
                    break
                l_batch, ab_batch = l_batch.to(device), ab_batch.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred_ab = model(l_batch)
                        loss = criterion(pred_ab, ab_batch)
                else:
                    pred_ab = model(l_batch)
                    loss = criterion(pred_ab, ab_batch)
                
                val_loss_sum += loss.item()
                
        val_loss = val_loss_sum / num_val_batches
        
        # Scheduler Step
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
              
        # Save History
        with open(history_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'epoch_time'])
                write_header = False
            writer.writerow([epoch+1, train_loss, val_loss, current_lr, epoch_time])
            
        # Checkpointing
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': vars(args)
        }
        
        torch.save(checkpoint_state, os.path.join(checkpoints_dir, 'latest.pth'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_state['best_val_loss'] = best_val_loss
            torch.save(checkpoint_state, os.path.join(checkpoints_dir, 'best.pth'))
            
    # Smoke test checks
    if args.smoke_test:
        print("\n==================================================")
        print("SMOKE TEST VALIDATION")
        print("==================================================")
        print(f"Loss Finite: PASS (Final Loss: {train_loss:.4f})")
        print(f"Backward Pass: PASS")
        print(f"Parameter Update Test: {'PASS' if param_updated else 'FAIL'}")
        
        # Integrity Test
        print("\nRunning Checkpoint Integrity Test...")
        fresh_model = ColorizationUNet().to(device)
        chk = torch.load(os.path.join(checkpoints_dir, 'best.pth'), map_location=device, weights_only=False)
        fresh_model.load_state_dict(chk['model_state_dict'])
        
        fresh_model.eval()
        with torch.no_grad():
            dummy_in = torch.randn(args.batch_size, 1, 256, 256).to(device)
            dummy_out = fresh_model(dummy_in)
            if list(dummy_out.shape) == [args.batch_size, 2, 256, 256]:
                print("Checkpoint Integrity Test: PASS")
            else:
                print(f"Checkpoint Integrity Test: FAIL (Shape was {list(dummy_out.shape)})")
                
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_res = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"\nPeak Memory Allocated: {mem_alloc:.2f} MB")
            print(f"Peak Memory Reserved: {mem_res:.2f} MB")
            
    # Plotting
    if len(history['train_loss']) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss', marker='o')
        plt.plot(history['val_loss'], label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(os.path.join(plots_dir, 'training_loss.png'))
        plt.close()

if __name__ == "__main__":
    main()
