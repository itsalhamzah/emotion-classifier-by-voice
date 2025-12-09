"""
Training script for emotion classification model - GPU OPTIMIZED
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import *
from model import EmotionCNN, count_parameters
from dataset import create_data_loaders

# Enable GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("✅ GPU optimizations enabled")

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

class Trainer:
    """
    Trainer class for emotion classification model
    """

    def __init__(self, model, train_loader, val_loader, device=DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        print(f"\n🖥️  Training Device: {device}")
        if device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Clear GPU cache periodically
            if self.device.type == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        # Final GPU cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Clear GPU cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_preds

    def train(self, epochs=EPOCHS):
        """Complete training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc, precision, recall, f1, _, _ = self.validate()
            self.val_losses.append(val_acc)
            self.val_accs.append(val_acc)

            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            if old_lr != new_lr:
                print(f"📉 Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            # GPU memory info
            if self.device.type == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model("best_model.pth")
                print("✓ Saved best model (lowest validation loss)")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model("best_acc_model.pth")
                print("✓ Saved best accuracy model")

        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print("="*60)

    def save_model(self, filename):
        """Save model checkpoint with versioning"""
        Path("models").mkdir(exist_ok=True)

        # Save with timestamp for version control
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_name = filename.replace('.pth', f'_{timestamp}.pth')
        versioned_path = os.path.join("models", versioned_name)

        # Also save with standard name (latest)
        standard_path = os.path.join("models", filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }

        # Save both versions
        torch.save(checkpoint, standard_path)
        torch.save(checkpoint, versioned_path)

        print(f"Model saved to {standard_path}")
        print(f"Version saved to {versioned_path}")

    def plot_training_history(self, save_path="training_history.png"):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.val_accs, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
        plt.close()


def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    """Main training function"""
    print("=" * 60)
    print("EMOTION CLASSIFICATION TRAINING")
    print("=" * 60)

    # Create data loaders
    print("\nLoading dataset...")
    try:
        train_loader, val_loader, test_loader, emotion_to_idx = create_data_loaders(
            DATASET_PATH, BATCH_SIZE
        )
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return

    # Create model
    print("\nInitializing model...")
    model = EmotionCNN(num_classes=NUM_CLASSES)

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Train model
    try:
        trainer.train(epochs=EPOCHS)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted!")
        trainer.save_model("interrupted_model.pth")
        return

    # Save final model
    trainer.save_model("final_model.pth")
    trainer.plot_training_history()

    # Final validation
    print("\nFinal validation:")
    val_loss, val_acc, precision, recall, f1, labels, preds = trainer.validate()
    print(f"Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    plot_confusion_matrix(labels, preds)

    print("\n✅ Training completed! Run: python gui_inference.py")


if __name__ == "__main__":
    main()