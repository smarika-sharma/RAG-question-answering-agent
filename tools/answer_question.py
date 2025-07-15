import os
import time
import smtplib
from langchain.tools import tool
from email.mime.text import MIMEText
from database.db_conn import SessionLocal, Booking
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from pinecone import Pinecone
from services.redis_service import redis_service
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

load_dotenv()

# Initialize the same model used for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

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


def retrieve_and_answer(query: str, top_k: int = 2, session_id: str = "default") -> str:
    """
    Query embedding and similarity search in pinecone, retrieve chunk_id, use chunk_id to retrieve full text from postgres with Redis caching for improved performance.
    
    Args:
        query: The user's question
        top_k: Number of top chunks to retrieve
        session_id: id for each user or session
        
    Returns:
        Answer to the question based on the retrieved chunk from postgres and llm 
    """
    
    try:
        # check in redis
        cached_response = redis_service.get_cached_response(query)
        if cached_response:
            return f"[CACHED] {cached_response['response']}"
        
        # Check for similar cached queries
        similar_responses = redis_service.find_similar_cached_queries(query, threshold=0.85)
        if similar_responses:
            best_similar = similar_responses[0]
            if best_similar['similarity'] > 0.9:
                return f"[SIMILAR CACHED] {best_similar['response']}"
        
        # Compute query embedding
        query_embedding = model.encode([query])[0]
        
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        print("Search results: ", results)
        
        if not results.matches:  # type: ignore
            response = "I couldn't find any relevant information to answer your question. Please make sure you have uploaded some documents first."
            redis_service.cache_response(query, response, 0.0)
            return response
        
        # Get the best match
        best_match = results.matches[0]  # type: ignore
        similarity_score = best_match.score
        chunk_id = best_match.metadata.get('chunk_uuid', '')

        # Log the similarity score and chunk id
        print(f"Similarity score={similarity_score:.2f}")
        print(f"Chunk id: {chunk_id}")

        # retrieve chunk from postgres data
        chunk_text = get_full_text_chunk(chunk_id)
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            api_key=SecretStr(api_key)
        )
        
        # Create a prompt for the LLM to generate a proper response
        prompt = f"""
        Based on the following information retrieved from documents, please provide a helpful and accurate answer to the user's question.
        
        User's question: {query}
        
        Retrieved information: {chunk_text}
        
        Please provide a clear, helpful response that directly addresses the user's question using the retrieved information. 
        """
        
        # Generate response using LLM
        llm_response = llm.invoke(prompt)
        response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Ensure response is a string
        if isinstance(response_content, list):
            response = " ".join(str(item) for item in response_content)
        else:
            response = str(response_content)
        
        # Cache the response
        redis_service.cache_response(query, response, similarity_score)
        
        # print(f"Response generated in: {time.time() - start_time:.3f}s")
        return response
        
    except Exception as e:
        error_response = f"Sorry, I encountered an error while trying to answer your question: {str(e)}"
        redis_service.cache_response(query, error_response, 0.0)
        return error_response

def get_full_text_chunk(chunk_uuid: str) -> str:
    """
    Retrieve the full text of a chunk from PostgreSQL.
    
    Args:
        chunk_uuid: The chunk UUID to retrieve
        
    Returns:
        Full text of the chunk
    """

    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )

        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT chunk_text FROM chunks WHERE chunk_id = %s",
            (chunk_uuid,)
        )
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result[0] if result else ""
        
    except Exception as e:
        print(f"Error retrieving full text from PostgreSQL: {e}")
        return ""

@tool
def answer_from_documents(user_query: str, session_id: str = "default") -> str:
    """Answer a question from the documents by retrieving the most relevant documents and returning the answer"""
    return retrieve_and_answer(user_query, session_id=session_id)

@tool
def book_interview(name, email, date, time):
    """ Book an interview

    Args:
        name: The name of the person booking the interview
        email: The email of the person booking the interview
        date: The date of the interview
        time: The preferred time of the interview

    Returns:
        A confirmation message that the booking was successful
    """
    # Check environment variables
    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not smtp_email or not smtp_password:
        raise ValueError("SMTP_EMAIL and SMTP_PASSWORD environment variables must be set")
    
    # Store to database
    session = SessionLocal()
    booking = Booking(name=name, email=email, date=date, time=time)
    session.add(booking)
    session.commit()
    session.close()

    # Send email
    msg = MIMEText(f"Booking confirmed for {name} on {date} at {time}.")
    msg["Subject"] = "Booking Confirmation"
    msg["From"] = smtp_email
    msg["To"] = smtp_email

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(msg["From"], msg["To"], msg.as_string())

    return "Booking confirmed and email sent."

__all__ = ["book_interview", "answer_from_documents"]