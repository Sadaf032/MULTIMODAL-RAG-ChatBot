import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


documents = []


# Images folder
image_folder = "images"


# Read image descriptions
with open(
    "image_descriptions.txt",
    "r",
    encoding="utf-8"
) as f:
    content = f.read()


# Split descriptions
chunks = content.split("=" * 80)


for chunk in chunks:

    if chunk.strip():

        chunk_text = chunk.strip()


        # Find image name from description
        image_path = None

        for image in os.listdir(image_folder):

            if image.lower() in chunk_text.lower():

                image_path = os.path.join(
                    image_folder,
                    image
                )

                break


        documents.append(
            Document(
                page_content=chunk_text,

                metadata={
                    "image_path": image_path
                }
            )
        )


print(
    f"Total documents: {len(documents)}"
)


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create FAISS vector database
vectorstore = FAISS.from_documents(
    documents,
    embeddings
)


# Save index
vectorstore.save_local(
    "image_faiss_index"
)


print(
    "✅ Image embeddings created successfully!"
)