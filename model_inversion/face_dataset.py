import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):

    def __init__(self, root, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.training_files_names = []
        self.root_dir = root
        self.transform = transform

        folders = os.listdir(root)
        for folder in folders:
            if folder.startswith("s"):
                files = os.listdir(root + folder + "/")
                files = [root + folder + "/" + fil for fil in files]
                self.training_files_names.extend(files)

    def __len__(self):
        return len(self.training_files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        train_image = np.asarray(Image.open(self.training_files_names[idx])) / 255.0
        train_image = train_image.astype(np.double)
        label = int(self.training_files_names[idx].split("/")[-2][1:]) - 1
        return train_image, label


if __name__ == "__main__":
    train_set = FaceDataset("./data/faces/training/")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=8,  # shuffle=True,
        num_workers=8, pin_memory=True)

    x, y = next(iter(train_loader))
    print(x.shape)
    _ = 0
