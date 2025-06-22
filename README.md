# 📊 AutoLysis — Automated Goodreads Data Analysis & Storytelling

**AutoLysis** is a Python-based tool that performs end-to-end automated **Exploratory Data Analysis (EDA)** on Goodreads data. It generates **insightful visualizations** and creates a **storytelling-style narrative** using a Large Language Model (LLM). This project was created as a part of the **Tools for Data Science** course.

---

## 🔍 Project Overview

This project explores trends and patterns in a Goodreads dataset by:
- Automatically analyzing structure and quality of the data
- Visualizing numeric and categorical distributions
- Identifying correlations and missing values
- Generating a narrative report using GPT API

## 🚀 Usage

### 🔑 Step 1: Set Your API Token

Before running the script, export your `AIPROXY_TOKEN` environment variable:

```bash
export AIPROXY_TOKEN=your_openai_or_proxy_token
```

---

### ▶️ Step 2: Run the Script

Use the following command to start the analysis:

```bash
python autolysis.py goodreads/goodreads.csv
```

---

### ✅ This script will:

- 📥 Load and analyze the dataset  
- 📊 Save visualizations to the `visualizations/` folder  
- 🤖 Generate a narrative using GPT  
- 📝 Save that narrative to `README.md`

---

## 📈 Visualizations Output

All generated plots will be saved in the `visualizations/` directory:

- `missing_data_heatmap.png`
- `correlation_heatmap.png`
- `pairplot.png`
- `boxplot_<numeric>_by_<categorical>.png`
- `<column>_distribution.png`
