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

# Fetch API token from environment
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

def load_data(file_path):
    """Load CSV data with encoding detection."""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform basic data analysis."""
    try:
        numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
        # Use a fallback for datetime handling for older pandas versions
        try:
            summary = df.describe(include='all', datetime_is_numeric=True).to_dict()
        except TypeError:  # For older pandas versions
            summary = df.describe(include='all').to_dict()

        analysis = {
            'summary': summary,
            'missing_values': df.isnull().sum().to_dict(),
            'correlation': numeric_df.corr().to_dict()  # Compute correlation only on numeric columns
        }
        return analysis
    except Exception as e:
        print(f"Error during data analysis: {e}")
        sys.exit(1)

def visualize_data(df):
    """Generate and save visualizations."""
    sns.set(style="whitegrid")
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        print("No numeric columns found for visualization.")
        return
    
    try:
        for column in numeric_columns:
            plt.figure()
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(f'{column}_distribution.png')
            plt.close()
        print("Visualizations saved successfully.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

def generate_narrative(analysis):
    """Generate narrative using LLM."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(API_URL, headers=headers, json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return "Narrative generation failed due to an error."

def main(file_path):
    df = load_data(file_path)
    analysis = analyze_data(df)
    visualize_data(df)
    narrative = generate_narrative(analysis)
    
    # Save the narrative to README.md
    try:
        with open('README.md', 'w') as f:
            f.write(narrative)
        print("Narrative saved to README.md.")
    except Exception as e:
        print(f"Error writing narrative to file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])

