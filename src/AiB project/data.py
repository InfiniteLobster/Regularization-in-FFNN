#--------------------Libraries--------------------#
import os
import numpy as np
#--------------------Constants--------------------#
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

#--------------------Functions--------------------#
def encode_peptide(seq, amino_acids=AMINO_ACIDS):
    encoding = np.zeros(len(seq) * len(amino_acids), dtype=np.float32)
    for pos, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            encoding[pos * len(amino_acids) + AA_TO_IDX[aa]] = 1.0
    return encoding
#Basic data loading handler.
def list_non_dat_files(root):
    files = []
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".dat"):
                files.append(os.path.join(folder_path, fname))
    return sorted(files)

class CustomDataset:
    def __init__(self, root):
        self.files = list_non_dat_files(root)
        self.X = []
        self.y = []
        for path in self.files:
            arr = np.genfromtxt(
                path,
                dtype=[("col1", "U50"), ("col2", "f8"), ("col3", "U50")],
                delimiter=None,
                encoding="utf-8"
            )
            arr = np.atleast_1d(arr)
            for row in arr:
                self.X.append(encode_peptide(row["col1"]))
                self.y.append(row["col2"])

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataLoader:
    def __init__(self, dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        for start in range(0, len(idxs), self.batch_size):
            batch_idxs = idxs[start:start + self.batch_size]
            batch = [self.dataset[i] for i in batch_idxs]
            yield batch