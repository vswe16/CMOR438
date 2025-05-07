import numpy as np
from airbnb_pricing_package.decision_tree_pricer import DecisionTreePricer

def test_decision_tree_predict():
    # Simple synthetic test data
    X = np.array([[1], [2], [3], [4]])
    y = np.array(["low", "low", "high", "high"])

    model = DecisionTreePricer(max_depth=2)
    model.train(X, y)

    pred = model.predict([[1.5]])
    assert pred[0] in ["low", "high"]
