# Health Risk Prediction & Insight System

This project presents a machine learning system designed to analyze patient data and predict health risk, similar to a simplified digital health analytics module. The system not only predicts whether a patient is at high or low risk but also provides interpretable insights to explain the prediction.
A synthetic dataset was generated to simulate realistic healthcare conditions, and the system integrates data analysis, machine learning, and visualization using Power BI.

# Dataset
A synthetic dataset of 300 patients was generated using a custom Python class. Each patient record includes the following features:
- Age
- Gender
- Blood Pressure
- Heart Rate
-	BMI
-	Glucose Level
-	Smoking Status
-	Activity Level
- Sleep Hours

A risk score was manually defined based on medical-like rules (e.g., high BMI, high glucose, low activity ...), and a binary target variable (risk) was created: 
•	0 → Low Risk 
•	1 → High Risk 

# Exploratory Data Analysis (EDA)
Exploratory analysis was performed to understand the data distribution and relationships between variables. 
Since the dataset was synthetically generated using predefined rules, the observed patterns reflect the logic used during data creation.

# Machine Learning Model (ML)
Several machine learning models were tested, and Random Forest was selected as the final model due to its superior performance.
The dataset was split using a train-test split approach to ensure proper evaluation.

# Model Performance
•	Accuracy: 0.83 
•	Precision: 0.87 
•	Recall: 0.82 

# Health Insight Generation 
In addition to prediction, the system includes a rule-based explanation module that generates interpretable insights. 
Example output: 
"High risk due to elevated BMI and high glucose level, combined with low physical activity."


# Conclusion 
This project demonstrates a complete pipeline for health risk prediction, including data generation, analysis, model training, and visualization. 
The system successfully: 
•	Predicts patient risk using machine learning 
•	Provides interpretable explanations for predictions 
•	Visualizes key patterns through an interactive dashboard 



