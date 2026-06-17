from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.encoders = {}

    def preprocess(self):
        df = self.df.copy()

        categorical_cols = df.select_dtypes(include="object").columns

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        X = df.drop("risk", axis=1)
        y = df["risk"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def train(self):
        X_train, X_test, y_train, y_test = self.preprocess()
        self.model.fit(X_train, y_train)
        return self.model, X_test, y_test, self.encoders