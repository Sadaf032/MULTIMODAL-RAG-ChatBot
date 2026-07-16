import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Gemini model
model = genai.GenerativeModel("gemini-flash-latest")

# Streamlit page config
st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🖼️"
)

# Title
st.title("🖼️ Multimodal RAG Chatbot")
st.write("Ask questions about your images and diagrams")

# Load FAISS Vector Database
@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "image_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

vectorstore = load_vectorstore()

# User Query
query = st.text_input(
    "Enter your question:"
)

if query:

    # Retrieve relevant documents/images
    results = vectorstore.similarity_search(
        query,
        k=3
    )

    # Combine context
    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    # Gemini Prompt
    prompt = f"""
You are a helpful Multimodal AI assistant.

Use the following image descriptions to answer the question.

Image Context:
{context}

Question:
{query}

Give a clear and accurate answer.
"""

    try:
        # Generate response
        response = model.generate_content(prompt)

        # Answer section
        st.subheader("🤖 Answer")
        st.write(response.text)

    except Exception as e:
        st.error(f"Gemini API Error:\n\n{e}")

    # Retrieved images section
    st.subheader("🔎 Retrieved Image Context")

    for doc in results:

        st.write(doc.page_content)

        # Display image if path exists in metadata
        image_path = doc.metadata.get("image_path")

        if image_path:

            if os.path.exists(image_path):
              st.image(
             image_path,
             caption=os.path.basename(image_path),
            width="stretch"
              )

            else:

                st.warning(
                    f"Image not found: {image_path}"
                )

        st.divider()