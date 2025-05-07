# decision_tree_pricer.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

class DecisionTreePricer:
    def __init__(self, max_depth=None, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def train(self, X_train, y_train):
        """
        Fit the Decision Tree classifier.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions on new data.
        """
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance.
        Returns accuracy and classification report.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return acc, report

    def get_feature_importances(self, feature_names=None):
        """
        Return feature importances from the trained tree.
        """
        importances = self.model.feature_importances_
        if feature_names:
            return dict(zip(feature_names, importances))
        return importances

    def export_tree(self, feature_names, class_names, filename="tree.dot"):
        """
        Export the tree to a DOT file (for visualization in Graphviz).
        """
        from sklearn.tree import export_graphviz
        with open(filename, "w") as f:
            export_graphviz(self.model, out_file=f,
                            feature_names=feature_names,
                            class_names=class_names,
                            filled=True, rounded=True,
                            special_characters=True)

