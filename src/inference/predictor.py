import torch
from PIL import Image
from torchvision import transforms


class Predictor:
    def __init__(self, model, weights_path, transform, class_names, device):
        self.model = model.to(device)
        self.device = device
        self.transform = transform
        self.class_names = class_names

        checkpoint = torch.load(weights_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        pil_image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)

        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(self.device)

    def predict(self, image_path: str) -> dict:
        image = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        predicted_class = self.class_names[predicted_idx.item()]

        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence.item())
        }

    def predict_top_k(self, image_path: str, k: int = 3) -> dict:
        image = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            top_probabilities, top_indices = torch.topk(probabilities, k=k, dim=1)

        predicted_idx = top_indices[0][0].item()
        predicted_class = self.class_names[predicted_idx]
        confidence = float(top_probabilities[0][0].item())

        top_k_results = []
        for class_idx, score in zip(top_indices[0], top_probabilities[0]):
            class_name = self.class_names[class_idx.item()]
            top_k_results.append((class_name, float(score.item())))

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_k": top_k_results
        }