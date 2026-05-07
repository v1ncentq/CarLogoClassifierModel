from torch.utils.data import Dataset
from PIL import Image

class CarLogoDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of tuples (image_path, label)
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, label