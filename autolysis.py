#/// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "chardet",
# ]
# ///
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool

# Constants
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

def load_data(file_path):
    """Load CSV data with encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result.get('encoding', 'utf-8')
        print(f"Detected encoding: {encoding}")
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform advanced data analysis."""
    try:
        numeric_df = df.select_dtypes(include=['number'])
        scaled_numeric = MinMaxScaler().fit_transform(numeric_df) if not numeric_df.empty else []
        if not numeric_df.empty:
            isolation_forest = IsolationForest(contamination=0.05, random_state=42)
            anomalies = isolation_forest.fit_predict(scaled_numeric)
            df['Anomaly_Score'] = anomalies

        analysis = {
            'shape': df.shape,
            'columns': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': df.nunique().to_dict(),
            'summary': df.describe(include='all').to_dict(),
            'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {},
            'anomaly_summary': df['Anomaly_Score'].value_counts().to_dict() if 'Anomaly_Score' in df else "N/A"
        }
        return analysis
    except Exception as e:
        print(f"Error during data analysis: {e}")
        sys.exit(1)

def plot_violin(column, df, output_dir):
    """Generate violin plots for numeric columns."""
    try:
        plt.figure()
        sns.violinplot(x=df[column], color='cyan')
        plt.title(f'Violin Plot of {column}')
        plt.savefig(f"{output_dir}/{column}_violin.png")
        plt.close()
    except Exception as e:
        print(f"Error plotting violin plot for {column}: {e}")

def visualize_data(df, output_dir="visualizations"):
    """Generate and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Sample the dataset for faster visualization
    sample_df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
    numeric_columns = sample_df.select_dtypes(include=['number']).columns

    try:
        # Visualize missing data pattern
        plt.figure(figsize=(10, 6))
        sns.heatmap(sample_df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Data Heatmap")
        plt.savefig(f"{output_dir}/missing_data_heatmap.png")
        plt.close()

        # Parallelize numeric column violin plots
        with Pool(processes=4) as pool:
            pool.starmap(plot_violin, [(col, sample_df, output_dir) for col in numeric_columns])

        # Correlation heatmap for a subset of numeric columns
        if len(numeric_columns) > 1:
            sampled_columns = numeric_columns[:10]  # Limit the number of columns
            plt.figure(figsize=(10, 8))
            corr_matrix = sample_df[sampled_columns].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.savefig(f"{output_dir}/correlation_heatmap.png")
            plt.close()

        print("Visualizations saved successfully.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

def generate_narrative(analysis):
    """Generate a storytelling-style narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = (
        f"You are a data analyst detective solving mysteries. Here is the data analysis report:\n\n"
        f"Dataset Summary:\n"
        f"Shape: {analysis['shape']}\n"
        f"Columns: {analysis['columns']}\n\n"
        f"Missing Data:\n"
        f"{analysis['missing_values']}\n\n"
        f"Unique Values:\n"
        f"{analysis['unique_values']}\n\n"
        f"Summary Statistics:\n"
        f"{analysis['summary']}\n\n"
        f"Correlation Insights:\n"
        f"{analysis['correlation']}\n\n"
        f"Anomaly Detection Results:\n"
        f"{analysis['anomaly_summary']}\n\n"
        f"Write a detailed and engaging report as if unraveling the story of the data. Highlight trends, correlations, anomalies, and potential insights. Use a storytelling tone."
    )
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        content = response.json()
        return content.get('choices', [{}])[0].get('message', {}).get('content', "Narrative generation failed.")
    except Exception as e:
        print(f"Error during narrative generation: {e}")
        return "Narrative generation failed due to an error."

def save_narrative(narrative, file_path="README.md"):
    """Save narrative to a file."""
    try:
        with open(file_path, 'w') as f:
            f.write("# Dataset Detective Report\n\n")
            f.write(narrative)
        print(f"Narrative saved to {file_path}.")
    except Exception as e:
        print(f"Error writing narrative to file: {e}")

def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)
    narrative = generate_narrative(analysis)
    save_narrative(narrative)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
