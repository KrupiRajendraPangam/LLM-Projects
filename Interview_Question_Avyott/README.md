# Image-Based Table Extraction and Query Tool

This project provides an intelligent web-based interface to extract tables from images using OCR and answer natural language questions about the extracted table using a local LLaMA 2 language model.

---

# Llama 2

Llama 2 is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. This is the repository for the 7B fine-tuned model, optimized for dialogue use cases and converted for the Hugging Face Transformers format. Links to other models can be found in the index at the bottom.

---

## 🚀 Features

- 📸 Upload an image with a table (PNG/JPG/JPEG)
- 🔍 Extract table text using **Tesseract OCR**
- 📊 Parse and display the table as a **DataFrame**
- 📁 Download the table as a **PDF**
- 🤖 Ask natural language questions about the extracted data using **LLaMA 2**
- 🧮 Smart reasoning and explanations (limited math capability)

## 🧪 Example Use Case

Upload the image

---

> ![Streamlit App Screenshot](app_screenshot/Image_01.png)

---

Extract the text using OCR

---

> ![Streamlit App Screenshot](app_screenshot/Image_02.png)

---

Convert into data frame(Download PDF file or csv file)

---

> ![Streamlit App Screenshot](app_screenshot/Image_03.png)

---

Extracted text given as input to LLaMa 2

---

> ![Streamlit App Screenshot](app_screenshot/Image_04.png)

---
