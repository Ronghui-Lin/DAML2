import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import time

# Import from local
from model import AircraftDetector
from dataset import AircraftDataset, collate_fn, get_transform
from loss import calculate_loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    reg_loss_accum = 0.0
    processed_batches = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Training]")

    start_time = time.time()
    for i, (images, targets) in enumerate(progress_bar):
        # Handle potential None batch from collate_fn if all images failed
        if images is None or targets is None:
            print(f"Warning: Skipping batch {i} due to image loading errors.")
            continue

        images = images.to(device)
        # Targets remain a list of dicts, loss function handles moving boxes to device

        # Forward pass
        cls_logits, box_preds = model(images)

        # Calculate loss
        loss, cls_loss, reg_loss = calculate_loss(cls_logits, box_preds, targets,
                                                  model_stride=32, # Match AircraftDetector stride
                                                  cls_weight=1.0, reg_weight=1.0) 

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss encountered at batch {i}. Skipping batch.")
            continue

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss_accum += loss.item()
        cls_loss_accum += cls_loss.item()
        reg_loss_accum += reg_loss.item()
        processed_batches += 1

        # Update progress bar
        if (i + 1) % print_freq == 0 or i == len(data_loader) - 1:
             progress_bar.set_postfix({
                 'Loss': f'{total_loss_accum / processed_batches:.4f}',
                 'Cls': f'{cls_loss_accum / processed_batches:.4f}',
                 'Reg': f'{reg_loss_accum / processed_batches:.4f}'
             })

    end_time = time.time()
    epoch_time = end_time - start_time
    avg_loss = total_loss_accum / processed_batches if processed_batches > 0 else 0
    avg_cls_loss = cls_loss_accum / processed_batches if processed_batches > 0 else 0
    avg_reg_loss = reg_loss_accum / processed_batches if processed_batches > 0 else 0

    print(f"Epoch {epoch+1} [Training] Completed in {epoch_time:.2f}s: Avg Loss: {avg_loss:.4f}, Avg Cls Loss: {avg_cls_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    model.eval()
    total_loss_accum = 0.0
    cls_loss_accum = 0.0
    reg_loss_accum = 0.0
    processed_batches = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1} [Validation]")

    start_time = time.time()
    for i, (images, targets) in enumerate(progress_bar):
        if images is None or targets is None:
            print(f"Warning: Skipping validation batch {i} due to image loading errors.")
            continue

        images = images.to(device)

        # Forward pass
        cls_logits, box_preds = model(images)

        # Calculate loss
        loss, cls_loss, reg_loss = calculate_loss(cls_logits, box_preds, targets,
                                                  model_stride=32,
                                                  cls_weight=1.0, reg_weight=1.0)

        if torch.isnan(loss):
             print(f"Warning: NaN loss encountered during validation at batch {i}.")
             continue # Skip this batch for average calculation

        total_loss_accum += loss.item()
        cls_loss_accum += cls_loss.item()
        reg_loss_accum += reg_loss.item()
        processed_batches += 1

        # Update progress bar
        progress_bar.set_postfix({
             'Loss': f'{total_loss_accum / processed_batches:.4f}',
             'Cls': f'{cls_loss_accum / processed_batches:.4f}',
             'Reg': f'{reg_loss_accum / processed_batches:.4f}'
        })

    end_time = time.time()
    epoch_time = end_time - start_time
    avg_loss = total_loss_accum / processed_batches if processed_batches > 0 else 0
    avg_cls_loss = cls_loss_accum / processed_batches if processed_batches > 0 else 0
    avg_reg_loss = reg_loss_accum / processed_batches if processed_batches > 0 else 0

    print(f"Epoch {epoch+1} [Validation] Completed in {epoch_time:.2f}s: Avg Loss: {avg_loss:.4f}, Avg Cls Loss: {avg_cls_loss:.4f}, Avg Reg Loss: {avg_reg_loss:.4f}")
    return avg_loss


def main(args):
    # I'm using an RTX 4080 to do this task, training times may be longer on a worse CUDA supported GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Datasets
    dataset_train = AircraftDataset(csv_file=args.csv_path, root_dir=args.data_dir,
                                    split='train', transform=get_transform(train=True))
    dataset_val = AircraftDataset(csv_file=args.csv_path, root_dir=args.data_dir,
                                  split='validation', transform=get_transform(train=False))

    # Create DataLoaders
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    # Create Model
    # AircraftDetector is defined in model.py
    # Using resnet34 as default, num_classes=1 is hardcoded in model
    model = AircraftDetector(backbone_name='resnet34', pretrained_backbone=True)
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training Loop
    print("Starting training")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, args.print_freq)
        val_loss = evaluate(model, data_loader_val, device, epoch)

        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
             lr_scheduler.step(val_loss)

        #Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, f"best_model_epoch_{epoch+1}_loss_{val_loss:.4f}.pth")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, save_path)
            print(f"Saved best model to {save_path}")

        # Save Last Model
        last_save_path = os.path.join(args.output_dir, "last_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, last_save_path)


    print("Training complete")
    print(f"Best model achieved: {best_val_loss:.4f}")
    print(f"Model checkpoints saved in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple PyTorch Aircraft Detection Training")
    parser.add_argument('--data_dir', required=True, help='Path to the root directory of resized data that have train/val/test folder')
    parser.add_argument('--csv_path', required=True, help='Path to the resized_annotations.csv file')
    parser.add_argument('--output_dir', default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=20, type=int, help='no. of training epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--workers', default=6, type=int, help='no. of data loading workers')
    parser.add_argument('--print_freq', default=50, type=int, help='Frequency of printing training progress')

    args = parser.parse_args()

    main(args)