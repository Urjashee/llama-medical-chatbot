from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from src.pinecone import delete_pinecone_index, insert_or_fetch_embeddings

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

delete_pinecone_index()
vector_store = insert_or_fetch_embeddings(text_chunks, embeddings, index_name)
