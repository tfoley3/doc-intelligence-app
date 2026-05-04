import fitz 
import pytesseract
from PIL import Image
import io 
import os 

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(file_bytes):
    text = ""
    pdf = fitz.open(stream=file_bytes, filetype="pdf")
    
    for page in pdf:
        page_text = page.get_text()
        
        if page_text.strip():
            text += page_text
        else:
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            text += pytesseract.image_to_string(img)
    
    return text

def extract_text_from_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(img)
    return text

def process_document(file_bytes, filename):
    extension = os.path.splitext(filename)[1].lower()
    
    if extension == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        return extract_text_from_image(file_bytes)
    else:
        return "Unsupported file type."
    
    