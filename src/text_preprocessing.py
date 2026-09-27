import re
import io
import nltk
from nltk.corpus import stopwords
import pypdf

# Ensure NLTK stopwords dataset is available locally
try:
    _STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _STOP_WORDS = set(stopwords.words('english'))


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extracts text content from an uploaded PDF file stream or byte stream.
    
    Raises:
        ValueError: If file is empty, corrupted, or contains no extractable text.
    """
    if uploaded_file is None:
        raise ValueError("No file provided for extraction.")
        
    try:
        if isinstance(uploaded_file, bytes):
            stream = io.BytesIO(uploaded_file)
        elif hasattr(uploaded_file, 'read'):
            stream = io.BytesIO(uploaded_file.read())
        else:
            stream = uploaded_file

        reader = pypdf.PdfReader(stream)
        extracted_text = []

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text)

        full_text = "\n".join(extracted_text).strip()

        if not full_text:
            raise ValueError(
                "The uploaded PDF file contains no readable text. "
                "It may be an image-only/scanned document or password protected."
            )

        return full_text

    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Failed to process PDF file: {str(e)}")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocesses text for NLP matching:
    1. Lowercase conversion
    2. Removal of non-essential punctuation (preserving + and # for C++, C#)
    3. Whitespace normalization
    4. Optional stop-word removal
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase conversion
    cleaned = text.lower()

    # Preserve programming syntax punctuation like C++, C#
    # Replace weird bullet points or special characters with spaces
    cleaned = re.sub(r'[^\w\s\+\#\.-]', ' ', cleaned)

    # 3. Whitespace normalization
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 4. Stop-word removal if requested
    if remove_stopwords:
        tokens = cleaned.split()
        filtered_tokens = [t for t in tokens if t not in _STOP_WORDS]
        cleaned = " ".join(filtered_tokens)

    return cleaned
