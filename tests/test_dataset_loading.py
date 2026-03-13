from src.datasets.base_dataset import Dataset
from src.datasets.boolean_dataset_and import ANDDataset

def test_dataset_loading():

    path = 'data/test/and_test.csv'

    dataset_gen = ANDDataset(path=path)
    dataset_gen.generate()

    dataset = Dataset(path, target_column='class')

    X, y = dataset.load()

    assert X.shape[0] > 0
    assert y.shape[0] > 0