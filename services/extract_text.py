from PyPDF2 import PdfReader


def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_txt(file) -> str:
    try:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')  # decode if needed
        return content.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from TXT file: {str(e)}")
