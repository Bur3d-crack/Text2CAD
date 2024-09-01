
from torch.utils.data import Dataset

class TextPointCloudDataset(Dataset):
    def __init__(self, text_descriptions, point_clouds):
        self.text_descriptions = text_descriptions
        self.point_clouds = point_clouds

    def __len__(self):
        return len(self.text_descriptions)

    def __getitem__(self, idx):
        return self.text_descriptions[idx], self.point_clouds[idx]
