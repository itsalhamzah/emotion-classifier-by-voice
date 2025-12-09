"""
Fine-tune model with lower learning rate
"""
import torch
from train import Trainer
from model import EmotionCNN
from dataset import create_data_loaders
from config import *


def finetune_model():
    print("Loading data...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(DATASET_PATH, BATCH_SIZE)

    print("Loading existing model...")
    model = EmotionCNN(num_classes=NUM_CLASSES)
    checkpoint = torch.load("models/best_acc_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Fine-tuning with lower learning rate...")
    trainer = Trainer(model, train_loader, val_loader)

    # Lower learning rate for fine-tuning
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = 0.0001  # Much lower than original 0.001

    # Train for more epochs with careful learning
    trainer.train(epochs=30)

    trainer.save_model("finetuned_model.pth")
    trainer.plot_training_history("finetuned_history.png")

    print("\n✅ Fine-tuning completed!")


if __name__ == "__main__":
    finetune_model()