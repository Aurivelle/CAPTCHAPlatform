from torch.utils.data import Dataset
from PIL import Image
import os


class CharDataset(Dataset):
    def __init__(self, root_dir, transform=None, charset=None):
        """
        root_dir: 'dataset/'，裡面有 labels.txt 以及各張圖片
        charset: 字符表字串，例如 '0123456789abcdefghijklmnopqrstuvwxyz'
        """
        self.root = root_dir
        self.transform = transform
        # 建立字符到 index 的映射
        self.chars = charset
        self.char2idx = {c: i for i, c in enumerate(self.chars)}

        # 讀 labels.txt
        self.samples = []
        with open(os.path.join(root_dir, "labels.txt"), "r", encoding="utf-8") as f:
            for line in f:
                fn, ch = line.strip().split()
                path = os.path.join(root_dir, fn)
                if os.path.isfile(path):
                    self.samples.append((path, self.char2idx[ch]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
