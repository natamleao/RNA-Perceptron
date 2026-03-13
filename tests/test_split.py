from src.datasets.base_dataset import Dataset
from src.datasets.boolean_dataset_and import ANDDataset
from src.helpers.utils import train_test_split_manual

def test_train_test_split():

    path = 'data/test/and_test.csv'

    dataset_gen = ANDDataset(path=path)
    dataset_gen.generate()

    dataset = Dataset(path, target_column='class')
    X, y = dataset.load()

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_seed=42)

    assert len(X_train) > 0
    assert len(X_test) > 0