from data.generator import HealthDataGenerator
from analysis.eda_analyzer import EDAAnalyzer
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from insights.insight_generator import HealthInsightGenerator


def main():
    # 1) Generate dataset
    generator = HealthDataGenerator(n_samples=300)
    df = generator.save_dataset()

    # 2) EDA
    analyzer = EDAAnalyzer(df)
    analyzer.run_eda()

    # 3) Train one model (Random Forest)
    trainer = ModelTrainer(df)
    model, X_test, y_test, encoders = trainer.train()

    # 4) Evaluate model
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(X_test, y_test)
    evaluator.display_results(results)

    # 5) Generate health insight for a new patient
    patient = {
        "age": 60,
        "gender": "Male",
        "blood_pressure": 150,
        "heart_rate": 92,
        "bmi": 32.5,
        "glucose_level": 170,
        "smoking_status": "Current",
        "activity_level": "Low",
        "sleep_hours": 5
    }

    insight_generator = HealthInsightGenerator(model, encoders)
    insight = insight_generator.generate_insight(patient)

    print("\n=== Health Insight ===")
    print("Prediction :", insight["prediction"])
    print("Explanation:", insight["explanation"])


if __name__ == "__main__":
    main()