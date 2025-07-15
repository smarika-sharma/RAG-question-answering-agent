import os
import psycopg2
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from typing import List
from dotenv import load_dotenv
from datetime import datetime
import uuid

load_dotenv()

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
if not pinecone_index_name:
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set")

pc= Pinecone(api_key=pinecone_api_key)
index= pc.Index(pinecone_index_name)

# db variables validation
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

if not all([db_host, db_port, db_name, db_user, db_password]):
    raise ValueError("Database environment variables (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) are not set")


def generate_embeddings(text_chunks: List[str], file: str) -> str:
    """
    Generate embeddings and store metadata in PostgreSQL + vectors in Pinecone.
    """

    # Model definition
    model_name="all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # embedding generation
    embeddings = model.encode(text_chunks, show_progress_bar=True)

    document_id = str(uuid.uuid4())
    embedding_model = model_name
    chunking_method = "token"

    # upserting in pinecone
    pinecone_data = []

    for i, embedding in enumerate(embeddings):
        # Unique UUID for each chunk: it is same in pinecone and postgres db for each chunk for easy retrieval of text chunk
        chunk_uuid = str(uuid.uuid4())  
        chunk_text = text_chunks[i]
        
        # prepare data for upserting in pinecone
        pinecone_data.append((
            chunk_uuid,
            embedding.tolist(),
            {
                "document_id": document_id,
                "chunk_uuid": chunk_uuid,
                "chunk_index": i,
                "filename": file
            }
        ))

    # Upsert to Pinecone
    index.upsert(vectors=pinecone_data)
    print(f"Upserted {len(pinecone_data)} chunks for document {document_id} into Pinecone.")

    # Insert metadata into PostgreSQL
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        cursor = conn.cursor()

        for i, chunk_text in enumerate(text_chunks):
            # Get the same UUID that was used for this chunk in Pinecone
            chunk_uuid = pinecone_data[i][0]  # Get UUID from Pinecone data

            cursor.execute(
                """
                INSERT INTO chunks (chunk_id, document_id, chunk_index, chunk_text, embedding_model, chunking_method)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                (chunk_uuid, document_id, i, chunk_text, embedding_model, chunking_method)
            )

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print("Error inserting metadata into PostgreSQL:", str(e))

    return "successful"
