import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms


def partition_dataset(dataset, percentage_train, logs=True):
    # 1) Shuffle dataset
    s = np.random.shuffle(dataset)

    # Info: logs
    if logs:
        print('Dataset: ', dataset.shape)
        print('One data sample shape: ', dataset[0].shape)

    # 2) Make mask to make the white pixels white
    dataset[dataset > 0.9] = 1

    # 2) Get numbers for train and test
    num_train_samples = int(percentage_train * dataset.shape[0])
    train_set = dataset[:num_train_samples]
    test_set = dataset[num_train_samples:]
    if logs:
        print('\nDividing dataset into:')
        print('Train set: ', train_set.shape[0])
        print('Test set: ', test_set.shape[0])
    return train_set, test_set


# Class to prepare data to pytorch engine
class MyDataset(data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.X[index]
        img = np.pad(img, ((8, 8), (0, 0)), 'constant', constant_values=(1))
        img = torch.from_numpy(img).float()
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])(img)

        return img, 0