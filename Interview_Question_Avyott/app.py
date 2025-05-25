import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import re
from fpdf import FPDF
import io
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers


st.title("Image-Based Table Extraction and Query Tool")

uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

def df_to_pdf(dataframe):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    col_width = pdf.w / (len(dataframe.columns) + 1)
    row_height = pdf.font_size * 1.5

    # Header
    for col_name in dataframe.columns:
        pdf.cell(col_width, row_height, txt=str(col_name), border=1)
    pdf.ln(row_height)

    # Rows
    for _, row in dataframe.iterrows():
        for item in row:
            pdf.cell(col_width, row_height, txt=str(item), border=1)
        pdf.ln(row_height)

    pdf_output = pdf.output(dest='S').encode('latin1')
    pdf_buffer = io.BytesIO(pdf_output)
    return pdf_buffer

def query_llama_table(table_text, user_question):
    llm = CTransformers(
        model='models/llama-2-7b-chat.ggmlv3.q8_0',
        model_type='llama',
        config={'max_new_tokens':256, 'temperature':0.01}
    )

    prompt_template = """
You are given the extracted text from an image containing a product table:

{table_text}

Note:
- The text is extracted using OCR, so formatting issues like extra spaces, incorrect line breaks, or broken words can occur.
- Product names, brands, or other fields may include multiple words that are split due to OCR errors.
- Use your domain knowledge to intelligently reconstruct such entries where appropriate.

Follow these rules:
- Use standard math to answer numeric questions.
- Do not ignore any values unless the data is clearly corrupted.
- Do not guess or hallucinate; explain based only on provided information.
- When calculating values (e.g., average), show the actual steps and results.




Question: {user_question}

Answer:
"""




    prompt = PromptTemplate(
        input_variables=["table_text", "user_question"],
        template=prompt_template
    )

    formatted_prompt = prompt.format(table_text=table_text, user_question=user_question)
    response = llm(formatted_prompt)
    return response

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    ocr_text = pytesseract.image_to_string(image)
    st.text_area("Extracted Text", ocr_text, height=200)

    lines = ocr_text.strip().split('\n')
    rows = [re.split(r'\s+', line.strip()) for line in lines if line.strip()]

    if len(rows) > 1:
        try:
            df = pd.DataFrame(rows[1:], columns=rows[0])
        except Exception as e:
            st.error(f"Error creating DataFrame: {e}")
            df = None

        if df is not None:
            st.dataframe(df)

            pdf_data = df_to_pdf(df)
            st.download_button(
                label="📄 Download Table as PDF",
                data=pdf_data,
                file_name="extracted_table.pdf",
                mime="application/pdf"
            )

            user_query = st.text_input("Ask a question about the extracted table:")

            if user_query:
                with st.spinner("Querying LLaMA...."):
                    answer = query_llama_table(ocr_text, user_query)
                st.markdown("### LLaMA Answer:")
                st.write(answer)

    else:
        st.write("Could not detect table structure in the image text.")





