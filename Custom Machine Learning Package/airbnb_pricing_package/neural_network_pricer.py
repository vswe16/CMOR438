from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class NeuralNetworkPricer:
    def __init__(self, hidden_layer_sizes=(100,), max_iter=300, random_state=42):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                  max_iter=max_iter,
                                  random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        return {"mae": mae, "mse": mse}
