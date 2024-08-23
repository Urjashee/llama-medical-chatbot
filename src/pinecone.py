from langchain.vectorstores import Pinecone
import pinecone
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def delete_pinecone_index(index_name='all'):
    pc = pinecone.Pinecone()

    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes')
        for index in indexes:
            pc.delete_index(index)
        print('Done')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')


def insert_or_fetch_embeddings(text_chunks, embeddings, index_name):
    # importing the necessary libraries and initializing the Pinecone client
    from pinecone import ServerlessSpec
    pc = pinecone.Pinecone()

    # loading from existing index
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        vector_store = Pinecone.from_documents(documents=text_chunks, embedding=embeddings, index_name=index_name)
        print('Ok')

    return vector_store
