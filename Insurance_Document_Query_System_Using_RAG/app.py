import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import base64
import shutil
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from PIL import Image
import pytesseract

from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from dotenv import load_dotenv
import nest_asyncio

load_dotenv()
nest_asyncio.apply()

# Set up LlamaParse
llamaparse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
parser = LlamaParse(
    api_key=llamaparse_api_key,
    result_type="markdown"  
)

# ---- Config ----
pdfs_directory = os.path.join(os.getcwd(), "uploads")
os.makedirs(pdfs_directory, exist_ok=True) 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    # base_url='http://95.216.245.92:11434/'
)
model = OllamaLLM(
    model="llama3.2",
    # base_url='http://95.216.245.92:11434/'
)


# ---- Helper Functions ----
def display_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        return f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px" type="application/pdf"></iframe>'
    elif ext in [".jpg", ".jpeg", ".png"]:
        return f'<img src="data:image/{ext[1:]};base64,{base64.b64encode(open(file_path,"rb").read()).decode()}" width="100%" />'
    else:
        return "Preview not available for this file type."


def extract_text_from_pdf(pdf_file):
    """Extract full text using LlamaParse + isolate premium details for accuracy"""
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[pdf_file], file_extractor=file_extractor).load_data()
    full_text = "\n".join([doc.text for doc in documents])
    return full_text


def extract_text_from_image(image_file):
    img = Image.open(image_file)
    text = pytesseract.image_to_string(img)
    return text

def get_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        full_text = extract_text_from_pdf(file_path)
        return full_text
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_from_image(file_path), ""
    else:
        return "", ""


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", "|", ":", "."]
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, source_name="uploaded file"):
    metadatas = [{"source": source_name} for _ in text_chunks]
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)

    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = prompt_template ="""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )
    return create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )


def answer_question(user_question):
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.max_marginal_relevance_search(user_question, k=6)


    chain = get_conversational_chain()
    response = chain.invoke({
        "context": docs,
        "input": user_question
    })

    return response, docs


# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("Vehicle Insurance Document Chatbot")

st.markdown(
    "👋Welcome to the Vehicle Insurance Document Query bot!\n\n"
    "This bot helps you understand your vehicle insurance policies.\n\n"
    "You can upload your insurance PDF and then ask questions like:\n\n"
    "1. What is included and not included in the policy insurance?\n\n"
    "2. What is the policy number and the tenure of this policy?\n\n"
)

with st.sidebar:
    st.header("Upload your document")
    uploaded_file = st.file_uploader(
        "Upload PDF, JPG, PNG",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=False  
    )

    if uploaded_file:
        save_path = os.path.join(pdfs_directory, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown(f"**Filename:** {uploaded_file.name}")
        st.markdown(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        st.markdown("**File Preview:**", unsafe_allow_html=True)
        st.markdown(display_file(save_path), unsafe_allow_html=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing document..."):

                # If it's a NEW file, reset memory + FAISS
                if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
                    if os.path.exists("faiss_index"):
                        shutil.rmtree("faiss_index")
                    st.session_state.messages = []  # clear old chat
                    st.session_state.last_file = uploaded_file.name  # remember file

                # Extract text using LlamaParse (PDF) or Tesseract (Image)
                full_text = get_text_from_file(save_path)

                # Convert newlines to <br> before inserting into f-string
                text_html = full_text.replace("\n", "<br>")
                st.markdown(
                    f"<div style='max-height:300px;overflow-y:auto;border:1px solid #ddd;padding:10px;'>"
                    f"{text_html}"
                    f"</div>",
                    unsafe_allow_html=True
                )

                text_chunks = get_text_chunks(full_text)
                get_vector_store(text_chunks, source_name=uploaded_file.name)

                st.success("✅ Document processed! You can now ask questions.")

# ---- Chat Section ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            st.markdown("**Sources:**")
            for s in msg["sources"]:
                st.markdown(f"- 📄 {s}")

# Ask questions
if st.session_state.get("last_file") and os.path.exists("faiss_index"):
    question = st.chat_input("Ask about your policy...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        answer, docs = answer_question(question)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": list({d.metadata.get("source", "uploaded file") for d in docs})  
        })

        with st.chat_message("assistant"):
            st.write(answer)
            if docs:
                st.markdown("**Sources:**")
                for s in list({d.metadata.get("source", 'uploaded file') for d in docs}):
                    st.markdown(f"- 📄 {s}")