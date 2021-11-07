"""Definition of Dataloader"""

import numpy as np


def combine_batch_dicts(batch):
    batch_dict = {}
    for data_dict in batch:
        for key, value in data_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)
    return batch_dict


def batch_to_numpy(batch):
    numpy_batch = {}
    for key, value in batch.items():
        numpy_batch[key] = np.array(value)
    return numpy_batch


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):

        batch = []
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_iterator = iter(range(len(self.dataset)))
        for index in index_iterator:  # iterate over indices using the iterator
            batch.append(self.dataset[index])
            if (len(batch) == self.batch_size) or ((not self.drop_last) and (index == len(self.dataset) - 1)):
                yield batch_to_numpy(combine_batch_dicts(batch))
                batch = []

    def __len__(self):

        length = int(len(self.dataset) / self.batch_size)
        if not self.drop_last:
            length += 1

        return length
