from sklearn.metrics import accuracy_score, precision_score, recall_score


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        results = {
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "precision": round(precision_score(y_test, y_pred), 3),
            "recall": round(recall_score(y_test, y_pred), 3)
        }

        return results

    def display_results(self, results):
        print("\n=== Model Evaluation ===")
        print(f"Accuracy : {results['accuracy']}")
        print(f"Precision: {results['precision']}")
        print(f"Recall   : {results['recall']}")