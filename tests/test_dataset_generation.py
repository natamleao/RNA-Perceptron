import os
from src.datasets.boolean_dataset_and import ANDDataset

def test_dataset_generation():

    path = 'data/test/and_test.csv'

    dataset = ANDDataset(path=path)
    dataset.generate()

    assert os.path.exists(path)