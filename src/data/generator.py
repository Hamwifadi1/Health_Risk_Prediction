import pandas as pd
import numpy as np


class HealthDataGenerator:
    def __init__(self, n_samples: int = 200, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(self.random_state)

    def generate_base_features(self) -> pd.DataFrame:
        data = pd.DataFrame({
            "age": np.random.randint(18, 80, self.n_samples),
            "gender": np.random.choice(["Male", "Female"], self.n_samples),
            "blood_pressure": np.random.randint(90, 180, self.n_samples),
            "heart_rate": np.random.randint(55, 120, self.n_samples),
            "bmi": np.round(np.random.uniform(18, 40, self.n_samples), 1),
            "glucose_level": np.random.randint(70, 200, self.n_samples),
            "smoking_status": np.random.choice(["Non-smoker", "Former", "Current"], self.n_samples),
            "activity_level": np.random.choice(["Low", "Medium", "High"], self.n_samples),
            "sleep_hours": np.random.randint(4, 10, self.n_samples)
        })

        return data

    def compute_risk_score(self, data: pd.DataFrame) -> pd.Series:
        score = (
                (data["age"] > 50).astype(int) +
                (data["bmi"] > 30).astype(int) +
                (data["glucose_level"] > 140).astype(int) +
                (data["blood_pressure"] > 140).astype(int) +
                (data["smoking_status"] == "Current").astype(int) +
                (data["activity_level"] == "Low").astype(int) +
                (data["sleep_hours"] < 6).astype(int)
        )

        return score

    def assign_risk_label(self, score: pd.Series) -> pd.Series:
        return (score >= 3).astype(int)

    def generate_dataset(self) -> pd.DataFrame:
        data = self.generate_base_features()
        risk_score = self.compute_risk_score(data)
        data["risk"] = self.assign_risk_label(risk_score)
        return data

    def save_dataset(self, filepath: str = "data/health_data.csv") -> pd.DataFrame:
        data = self.generate_dataset()
        data.to_csv(filepath, index=False)
        return data
