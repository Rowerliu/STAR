from torch.utils.data import Dataset


class WSIDataset(Dataset):
    def __init__(self, df, wsi, transform, level=0, ps=256):
        # ps = patch size
        self.wsi = wsi
        self.transform = transform
        self.df = df
        self.level = level
        self.ps = ps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        offset = self.ps // 2
        # x，y分别对应原图上的横纵坐标
        x, y = self.df.iloc[idx, 0] - offset, self.df.iloc[idx, 1] - offset
        if x < 0: x = 0
        if y < 0: y = 0
        # patch = self.wsi.read_region((x, y), self.level, (self.ps, self.ps))
        patch = self.wsi.crop((x, y, x + self.ps, y + self.ps))
        patch = self.transform(patch.convert('RGB'))
        return patch
