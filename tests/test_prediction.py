import numpy as np
from src.models.perceptron_sk import PerceptronSK
from src.training.logger import TrainingLogger

def test_prediction_shape():

    X = np.array([
        [-1, -1],
        [ 1,  1]
    ])
    
    y = np.array([-1,1])

    logger = TrainingLogger(enabled=False)

    model = PerceptronSK(learning_rate=0.1, max_epochs=10, logger=logger)

    model.fit(X, y)

    preds = model.predict(X)

    assert len(preds) == 2