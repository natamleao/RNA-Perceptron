from src.preprocessing.scaler import StandardScaler 

def test_scaler():

    scaler = StandardScaler()

    X = [
        [1, 2],
        [3, 4],
        [5, 6]
    ]

    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == (3, 2)