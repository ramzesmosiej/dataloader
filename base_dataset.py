"""Dataset Base Class"""


class DummyDataset:
    """
    Simple dummy dataset
    Contains all integers from 1 to a given limit, which are dividable by a given divisor
    """

    def __init__(self, divisor, limit, **kwargs):
        """
        :param divisor: common divisor of all integers in the dataset
        :param limit: upper limit of integers in the dataset
        """
        super().__init__(**kwargs)
        self.data = [i for i in range(1, limit + 1) if i % divisor == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {"data": self.data[index]}
