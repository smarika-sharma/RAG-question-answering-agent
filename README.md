# RAG based Document Question Answering System with Interview Booking

A FastAPI application that present two APIs. 1. for file (pdf/txt) upload and 2. for booking and getting answer to the question from the uploaded document.

-Refer to Assessment_Documentation.pdf for outputs and detailed project workflow.

## Tech Stack

- **Backend**: FastAPI
- **Database**: PostgreSQL for metadata
- **Caching**: Redis
- **AI/ML**: Google Gemini for conversation with agent (llm), Sentence Transformers for embedding
- **Vector Database**: Pinecone

## Prerequisites

- Python 3.8+
- PostgreSQL
- Redis 
- Pinecone account
- Environment variables

## Installation

1. **Clone the repository**

2. **Create virtual environment and activate it**

3. **Install libraries**
   ```bash
   pip install -r requirements.txt
   ```

## Database Setup

### PostgreSQL Database

1. **Install PostgreSQL** 

2. **Start PostgreSQL service**
   ```bash
   sudo systemctl start postgresql
   ```

3. **Create database and user**
   ```sql
   -- Connect to PostgreSQL as superuser
   sudo -u postgres psql
   
   -- Create database
   CREATE DATABASE metadata;
   
   -- Create user
   CREATE USER <db_user> WITH PASSWORD '<db_password>';
   
   -- Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE metadata TO <db_user>;
   
   -- Connect to metadata database
   \c metadata
   
   -- Create chunks table
   CREATE TABLE chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID,
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunking_method TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );

   
   -- Create bookings table
   CREATE TABLE bookings (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    booking_date VARCHAR(100) NOT NULL,
    booking_time VARCHAR(50) NOT NULL
   );


   -- Grant table permissions
   GRANT ALL PRIVILEGES ON TABLE chunks TO <db_user>;
   GRANT ALL PRIVILEGES ON TABLE bookings TO <db_user>;
   
   -- Exit PostgreSQL
   \q
   ```

## Vector DB (Pinecone) Setup

1. **Create Pinecone index after logging in** 
   - Name: {any name: make sure to update in environment variables file}
   - Dimensions: `384`
   - Metric: `cosine`
3. **Get your API key** 

## Running the Application

1. **Start Redis** (optional, for better performance)
   ```bash
   redis-server
   ```

2. **Start the FastAPI application**
   ```bash
   uvicorn main:app --reload
   ```