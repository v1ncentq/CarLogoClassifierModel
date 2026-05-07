import torch
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_model(
    model,
    test_loader,
    criterion,
    device
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

    avg_loss = float(total_loss / len(test_loader))
    accuracy = float(correct / total) if total > 0 else 0.0

    precision = float(precision_score(all_labels, all_predictions, average="macro", zero_division=0))
    recall = float(recall_score(all_labels, all_predictions, average="macro", zero_division=0))
    f1 = float(f1_score(all_labels, all_predictions, average="macro", zero_division=0))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def collect_predictions(model, test_loader, device) -> tuple[list[int], list[int]]:
    model.eval()

    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

    return all_labels, all_predictions