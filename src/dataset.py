# src/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, file_paths, metadata, labels, transform=None):
        """
        Args:
            file_paths (list): List of image file paths.
            metadata (np.array): Array containing metadata for each image.
            labels (list or np.array): Labels corresponding to each image.
            transform: PyTorch transforms to be applied on the images.
        """
        self.file_paths = file_paths
        self.metadata = metadata
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        metadata = self.metadata[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # Convert metadata to tensor
        metadata = torch.tensor(metadata, dtype=torch.float)
        return image, metadata, label
