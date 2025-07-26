# ocr_pipeline.py
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm  # Install via pip if not already installed


# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_pdf_range(input_path, output_folder, start_page, end_page):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    os.makedirs(output_folder, exist_ok=True)
    for page_num in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_num])
    output_path = os.path.join(output_folder, f"extracted_pages_{start_page}_to_{end_page}.pdf")
    with open(output_path, "wb") as out_file:
        writer.write(out_file)
    return output_path

def extract_text_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    
    for page in tqdm(doc, desc="Extracting text with OCR", unit="page"):
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang='ben')
        all_text.append(text)
        
    return '\n'.join(all_text)

def run_ocr_pipeline(input_pdf, output_dir="resource/output_extracted"):
    os.makedirs("resource/raw_data", exist_ok=True)
    # Split into two parts
    passage_pdf = extract_pdf_range(input_pdf, output_dir, 2, 22)
    mcq_pdf = extract_pdf_range(input_pdf, output_dir, 23, 42)
    
    print("✅ PDF split into passage and MCQ sections.")
    
    # OCR each part
    passage_raw = extract_text_with_ocr(passage_pdf)
    mcq_raw = extract_text_with_ocr(mcq_pdf)

    # Save raw OCR results
    with open("resource/raw_data/passage_raw.txt", "w", encoding="utf-8") as f:
        f.write(passage_raw)
    with open("resource/raw_data/mcq_raw.txt", "w", encoding="utf-8") as f:
        f.write(mcq_raw)

    print("✅ Raw OCR text saved.")

if __name__ == "__main__":
    input_pdf_path = "resource/HSC26-Bangla1st-Paper.pdf"  # Path to your input PDF
    run_ocr_pipeline(input_pdf_path)
    print("✅ OCR pipeline completed successfully.")    
