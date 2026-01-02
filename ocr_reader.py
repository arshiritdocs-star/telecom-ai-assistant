# ocr_reader.py
import pytesseract
from PIL import Image
import os

# Optional: if using PDFs, install pdf2image
# pip install pdf2image
from pdf2image import convert_from_path

def ocr_image(image_path):
    """Read text from an image file using OCR."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def ocr_pdf(pdf_path):
    """Read text from a PDF file using OCR."""
    text = ""
    images = convert_from_path(pdf_path)
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

def ocr_folder(folder_path):
    """Read text from all images/PDFs in a folder."""
    all_text = ""
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            all_text += ocr_image(full_path) + "\n"
        elif filename.lower().endswith(".pdf"):
            all_text += ocr_pdf(full_path) + "\n"
    return all_text
