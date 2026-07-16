from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load FAISS database
vectorstore = FAISS.load_local(
    "image_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


def search_images(query, k=3):

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results



# Test query
query = "algorithm diagram"

results = search_images(query)


print("\nQuery:", query)
print("\nRetrieved Results:\n")


for doc in results:
    print(doc.page_content)
    print("-" * 80)