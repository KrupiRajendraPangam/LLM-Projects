#  Blog Generator using LLM

This is a simple web application built with **Streamlit** that generates blog content using a LLM (like GPT-2 or LLaMA) through `transformers`, `langchain`, and optionally It allows users to generate customized blog posts for specific audiences like **Researchers**, **Data Scientists**, or **Common People**.

---

## What is an LLM?

A **Large Language Model (LLM)** is an artificial intelligence model trained on vast amounts of textual data to understand and generate human-like text. LLMs like GPT, LLaMA, and others can perform tasks such as:

- Text generation
- Summarization
- Translation
- Question answering

In this project, we use an LLM locally to ensure fast, offline access and no dependency on APIs keys which usullay paid.

---

## Features

- Generate blog content by entering a topic
- Choose word count and target audience
- Run offline using a local LLM.
- Simple interface using Streamlit

---

## Setup Instructions

### 1. Clone the repo
```
git clone "your_repo_link"
cd your-repo
```
---

### 2. Create and activate a virtual environment
```
conda create -n bloggenv python=3.9
conda activate bloggenv
```
---

### 3. Install dependencies
```
pip install -r requirements.txt
```
---

### 4. Run the app
```
streamlit run app.py
```
---

### Acknowledgments

Special thanks to Krish Naik, whose tutorials and community-driven education made learning and building this LLM project possible.




