"""
Continue training from existing checkpoint
"""
import torch
from train import Trainer, main
from model import EmotionCNN
from dataset import create_data_loaders
from config import *


def continue_training(checkpoint_path, additional_epochs=20):
    """Continue training from saved checkpoint"""

    print("=" * 60)
    print(f"CONTINUING TRAINING FROM: {checkpoint_path}")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        DATASET_PATH, BATCH_SIZE
    )

    # Load model
    print("\nLoading existing model...")
    model = EmotionCNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader)

    # Load optimizer state
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore training history
    trainer.train_losses = checkpoint.get('train_losses', [])
    trainer.val_losses = checkpoint.get('val_losses', [])
    trainer.train_accs = checkpoint.get('train_accs', [])
    trainer.val_accs = checkpoint.get('val_accs', [])

    print(f"\nResuming training for {additional_epochs} more epochs...")
    print(f"Previous training: {len(trainer.train_losses)} epochs")

    # Continue training
    trainer.train(epochs=additional_epochs)

    # Save continued model
    trainer.save_model("continued_model.pth")
    trainer.plot_training_history("continued_training_history.png")

    print("\n✅ Continued training completed!")


if __name__ == "__main__":
    # Continue from best model
    continue_training("models/best_acc_model.pth", additional_epochs=20)