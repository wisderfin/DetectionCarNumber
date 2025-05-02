import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from utils import CHAR_TO_IDX, SIZE_IMAGE

transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(SIZE_IMAGE),
            transforms.ToTensor()
        ])

class ValidationDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        char_to_idx: dict = CHAR_TO_IDX,
        size: tuple[int, int] = SIZE_IMAGE,
        transform=None,
    ) -> None:
        self.img_dir = root_dir / 'img'
        self.ann_dir = root_dir / 'ann'
        self.char_to_idx = char_to_idx

        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size),
            transforms.ToTensor()
        ])

        self.samples = []
        for json_file in sorted(self.ann_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                img_name = data["name"]
                label = data["description"]

                img_path = self.img_dir / f"{img_name}.png"
                self.samples.append((img_path, label))

        print(f"Количество аннотаций: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = Image.open(img_path)
        image = self.transform(image)

        label_encoded = [self.char_to_idx[i] for i in label]

        return image, torch.tensor(label_encoded, dtype=torch.long)
