import torch
from pathlib import Path
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
        save_path,
        scheduler=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.scheduler = scheduler

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []

        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def train_one_epoch(self, epoch_index: int) -> dict[str, float]:
        self.model.train()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"[Train] Epoch {epoch_index + 1}/{self.num_epochs}",
            leave=False
        )

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

            current_loss = running_loss / total_samples
            current_acc = running_correct / total_samples

            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "acc": f"{current_acc:.4f}"
            })

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        epoch_accuracy = running_correct / total_samples if total_samples > 0 else 0.0

        return {
            "loss": float(epoch_loss),
            "accuracy": float(epoch_accuracy)
        }

    def validate_one_epoch(self, epoch_index: int) -> dict[str, float]:
        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        total_samples = 0

        progress_bar = tqdm(
            self.val_loader,
            desc=f"[Val]   Epoch {epoch_index + 1}/{self.num_epochs}",
            leave=False
        )

        with torch.no_grad():
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                preds = torch.argmax(outputs, dim=1)

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_correct += (preds == labels).sum().item()
                total_samples += batch_size

                current_loss = running_loss / total_samples
                current_acc = running_correct / total_samples

                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.4f}"
                })

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
        epoch_accuracy = running_correct / total_samples if total_samples > 0 else 0.0

        return {
            "loss": float(epoch_loss),
            "accuracy": float(epoch_accuracy)
        }

    def save_checkpoint(self, epoch: int, val_accuracy: float) -> None:
        output_path = Path(self.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "best_val_accuracy": val_accuracy,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(checkpoint, output_path)

    def fit(self) -> dict[str, list[float]]:
        print("\n" + "=" * 70)
        print("START TRAINING")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print("=" * 70)

        for epoch in range(self.num_epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate_one_epoch(epoch)

            train_loss = train_metrics["loss"]
            train_acc = train_metrics["accuracy"]
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            improved = val_acc > self.best_val_accuracy
            if improved:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint(epoch + 1, val_acc)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch + 1:02d}/{self.num_epochs:02d} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
                f"{'BEST' if improved else ''}"
            )

        print("=" * 70)
        print("TRAINING FINISHED")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Saved model: {self.save_path}")
        print("=" * 70 + "\n")

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_accuracy": self.train_accuracies,
            "val_accuracy": self.val_accuracies,
        }