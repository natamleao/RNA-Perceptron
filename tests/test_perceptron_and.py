from src.datasets.base_dataset import Dataset
from src.models.perceptron_sk import PerceptronSK
from src.preprocessing.scaler import StandardScaler 
from src.datasets.boolean_dataset_and import ANDDataset
from src.training.logger import TrainingLogger
from src.helpers.utils import train_test_split_manual

def test_perceptron_and_dataset():
    
    path = 'data/test/and_test.csv'

    dataset_gen = ANDDataset(path=path)
    dataset_gen.generate()

    dataset = Dataset(path, target_column='class')
    X, y = dataset.load()

    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_seed=42)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger = TrainingLogger(enabled=False)

    model = PerceptronSK(learning_rate=0.1, max_epochs=1000, logger=logger)

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # AND é linearmente separável
    assert accuracy >= 0.9