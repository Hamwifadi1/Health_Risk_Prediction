import pandas as pd


class HealthInsightGenerator:
    def __init__(self, model, encoders):
        self.model = model
        self.encoders = encoders

    def encode_patient(self, patient: dict) -> pd.DataFrame:
        patient_df = pd.DataFrame([patient])

        for col, encoder in self.encoders.items():
            patient_df[col] = encoder.transform(patient_df[col])

        return patient_df

    def generate_insight(self, patient: dict):
        encoded_patient = self.encode_patient(patient)
        prediction = self.model.predict(encoded_patient)[0]

        reasons = []

        if patient["bmi"] > 30:
            reasons.append("elevated BMI")

        if patient["glucose_level"] > 140:
            reasons.append("high glucose level")

        if patient["blood_pressure"] > 140:
            reasons.append("high blood pressure")

        if patient["activity_level"] == "Low":
            reasons.append("low physical activity")

        if patient["sleep_hours"] < 6:
            reasons.append("insufficient sleep")

        if patient["smoking_status"] == "Current":
            reasons.append("active smoking")

        if patient["age"] > 55:
            reasons.append("older age")

        if prediction == 1:
            if reasons:
                explanation = "High risk due to " + ", ".join(reasons) + "."
            else:
                explanation = "High risk."
        else:
            if reasons:
                explanation = "Low risk, but some factors should be monitored: " + ", ".join(reasons) + "."
            else:
                explanation = "Low risk with generally healthy indicators."

        if prediction == 1:
            risk_label = "High Risk"
        else:
            risk_label = "Low Risk"

        return {
            "prediction": risk_label,
            "explanation": explanation
        }

