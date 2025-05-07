import numpy as np
from airbnb_pricing_package.neural_network_pricer import NeuralNetworkPricer

def test_neural_network_predict():
    # Simple regression test
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([100, 150, 200, 250, 300])

    model = NeuralNetworkPricer(hidden_layer_sizes=(10,), max_iter=500)
    model.train(X, y)

    pred = model.predict([[3.5]])
    assert isinstance(pred[0], float), "Prediction should return a float price"
    assert 50 < pred[0] < 400, "Prediction seems out of expected price range"
