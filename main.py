from base_dataset import DummyDataset
from dataloader import DataLoader

simple_dataset = DummyDataset(2, 100)
data_loader = DataLoader(simple_dataset, batch_size=3, shuffle=True)
for batch in data_loader:
    print(batch)
