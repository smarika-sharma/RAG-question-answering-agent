from transformers import AutoTokenizer


def chunk_text_by_tokens(text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 200, overlap: int = 50):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer.encode(text)
    print(f"Total number of tokens: {len(tokens)}")
    
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i: i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens)
        chunks.append(chunk)
        i += chunk_size - overlap  
    
    return chunks
