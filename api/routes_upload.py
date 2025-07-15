from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from services.extract_text import extract_text_from_pdf, extract_text_from_txt
from services.chunk_text import chunk_text_by_tokens
from services.embed_store import generate_embeddings

from sentence_transformers import SentenceTransformer

router= APIRouter()

model = SentenceTransformer('all-MiniLM-L6-v2')


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):  #type: ignore
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    try:
        # extract text from pdf or txt file 
        if file.filename.endswith(".pdf"): #type: ignore
            text = extract_text_from_pdf(file.file)
        else:
            text = extract_text_from_txt(file.file)

        # chunk the extracted text 
        chunks = chunk_text_by_tokens(text)

        # generate embeddings
        embeddings=generate_embeddings(chunks, file.filename or "unknown")
        
        # the model returns embedding in 384 dimensions
        embedding_dims = 384

        return {
            "filename": file.filename,
            "num_chunks": len(chunks),
            "embedding_dims": embedding_dims,
            "status": "successfully embedded"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
