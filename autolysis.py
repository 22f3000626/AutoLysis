

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
    """Perform basic data analysis."""
    try:
        numeric_df = df.select_dtypes(include=['number'])
        analysis = {
            'shape': df.shape,
            'columns': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': df.nunique().to_dict(),
            'summary': df.describe(include='all').to_dict(),
            'correlation': numeric_df.corr().to_dict() if not numeric_df.empty else {}
        }
        return analysis
    except Exception as e:
        print(f"Error during data analysis: {e}")
        sys.exit(1)

def visualize_data(df, output_dir="visualizations"):
    """Generate and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns

    try:
        # Visualize missing data pattern
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Data Heatmap")
        plt.savefig(f"{output_dir}/missing_data_heatmap.png")
        plt.close()

        # Visualize numeric data distributions
        for column in numeric_columns:
            plt.figure()
            sns.histplot(df[column].dropna(), kde=True, color='blue')
            plt.title(f'Distribution of {column}')
            plt.savefig(f"{output_dir}/{column}_distribution.png")
            plt.close()

        # Pairplot for numeric columns
        if len(numeric_columns) > 1:
            pairplot = sns.pairplot(df[numeric_columns].dropna())
            pairplot.savefig(f"{output_dir}/pairplot.png")
            plt.close()

        # Correlation heatmap
        if len(numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_columns].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.savefig(f"{output_dir}/correlation_heatmap.png")
            plt.close()

        # Boxplots for numeric columns grouped by categorical variables
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for cat_col in categorical_columns:
            for num_col in numeric_columns:
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=df, x=cat_col, y=num_col)
                plt.title(f"Boxplot of {num_col} by {cat_col}")
                plt.xticks(rotation=45)
                plt.savefig(f"{output_dir}/boxplot_{num_col}_by_{cat_col}.png")
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
        f"Write a storytelling narrative based on the following data analysis summary:\n\n"
        f"Dataset Overview: \n"
        f"Shape: {analysis['shape']}\n"
        f"Columns: {analysis['columns']}\n\n"
        f"Missing Data: \n"
        f"{analysis['missing_values']}\n\n"
        f"Unique Values: \n"
        f"{analysis['unique_values']}\n\n"
        f"Summary Statistics: \n"
        f"{analysis['summary']}\n\n"
        f"Correlation Insights: \n"
        f"{analysis['correlation']}\n\n"
        f"Create a detailed narrative with analysis and insights that flow naturally like a report. "
        f"Explain trends, anomalies, and any correlations discovered. Use storytelling techniques to make it engaging."
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
