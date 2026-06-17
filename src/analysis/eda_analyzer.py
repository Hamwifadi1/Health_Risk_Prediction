from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        project_root = Path(__file__).resolve().parents[2]
        self.output_dir = project_root / "outputs" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def basic_info(self):
        print("\nDataset Shape:", self.df.shape)
        print("\nColumns:", self.df.columns.tolist())
        print("\nMissing Values:\n", self.df.isnull().sum())
        print("\nRisk Distribution:\n", self.df["risk"].value_counts())

    def plot_distributions(self):
        numerical_cols = self.df.columns

        for col in numerical_cols:
            if col == "risk":
                continue
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[col], kde=False)
            plt.title(f"Distribution of {col}")

            file_path = self.output_dir / f"{col}_distribution.png"
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()

    def plot_risk_distribution(self):
        plt.figure()
        sns.countplot(
            x="risk",
            data=self.df,
            palette={0: "green", 1: "red"}
        )
        plt.title("Risk Distribution (0 = Low Risk, 1 = High Risk)")
        plt.xlabel("Risk")
        plt.ylabel("Count")

        plt.savefig(self.output_dir / "risk_distribution.png")
        plt.close()

    def categorical_vs_risk(self):
        categorical_cols = self.df.select_dtypes(include="object").columns

        for col in categorical_cols:
            plt.figure()

            ax = sns.countplot(
                x=col,
                hue="risk",
                data=self.df,
                palette={0: "darkgreen", 1: "darkred"}
            )

            plt.title(f"{col} vs Risk")
            plt.xticks(rotation=30)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ["Low Risk", "High Risk"])
            plt.savefig(self.output_dir / f"{col}_vs_risk.png")
            plt.close()


    def correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        corr = self.df.corr()

        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")

        file_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()

        return corr

    def extract_key_patterns(self):
        grouped = self.df.groupby("risk").mean(numeric_only=True).round(2)
        return grouped

    def run_eda(self):
        self.basic_info()
        self.plot_distributions()
        self.plot_risk_distribution()
        self.categorical_vs_risk()
        corr = self.correlation_matrix()
        key_patterns = self.extract_key_patterns()

        print("\n=== Correlation Matrix ===\n")
        print(corr)

        print("\n=== Key Patterns ===\n")
        print(key_patterns)
