"""
Evaluate model performance
"""
import torch
from sklearn.metrics import classification_report
from dataset import create_data_loaders
from model import EmotionCNN
from config import *
from tqdm import tqdm


def evaluate_model():
    """Evaluate the trained model on test data"""
    print("Loading model...")
    model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load("models/best_acc_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Loading test data...")
    _, _, test_loader, _ = create_data_loaders(DATASET_PATH, BATCH_SIZE)

    print("Evaluating...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))
    print("=" * 60)

    # Calculate per-emotion accuracy
    print("\nPer-Emotion Accuracy:")
    print("-" * 60)
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(all_labels, all_preds)
    for i, emotion in enumerate(EMOTIONS):
        accuracy = cm[i, i] / cm[i].sum() * 100
        print(f"{emotion:10s}: {accuracy:6.2f}%")

    overall_accuracy = np.trace(cm) / np.sum(cm) * 100
    print("-" * 60)
    print(f"{'Overall':10s}: {overall_accuracy:6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_model()