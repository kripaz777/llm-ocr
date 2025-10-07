# Document Intelligence App (Streamlit + Vision LLM + OCR + Table Extraction)

A powerful **AI-based document analysis application** built with **Streamlit**, integrating **Vision LLMs (LLaVA / Qwen2-VL)**, **OCR**, and **PDF/Table parsing** tools like **pdfplumber** and **Camelot**.  
This project allows users to upload **images or PDF documents (invoices, bills, receipts, cheques, etc.)**, automatically detect and crop documents, and extract structured data (JSON format).

---

## Features

### Image Processing
- **Automatic Document Detection & Cropping** using `document-scanner-sdk`.
- **Manual Cropping (Optional)** via `streamlit-cropper` for precise selection (like CamScanner).
- **Preprocessing**: resizing, normalization, and noise reduction for better OCR and LLM performance.
- **OCR Fallback** using `pytesseract` when LLM vision extraction fails.

### PDF Processing
- Extracts tables and text using **pdfplumber** and **Camelot**.
- Handles **multi-page PDFs** and combines tables automatically.
- Detects duplicate columns and resolves them safely.
- Converts extracted data to **clean JSON output**.

### Vision LLM Integration
- Supports **offline models** such as **LLaVA (Ollama)** or **Qwen2-VL**.
- Multiple **document-type-specific prompts**:
  - Invoice / Bill
  - Bank Cheque
  - ID Document
  - Generic Form
- Returns **structured JSON** output for every extraction case.

### Intelligent Output Handling
- Automatically detects malformed or unstructured LLM output.
- Converts any valid key-value text response into **standardized JSON** format.
- Fallback logic for all failure cases (OCR → LLM → Text).

### User-Friendly Interface
- Drag-and-drop upload for images or PDFs.
- Dropdown to select document type (e.g., Invoice, Cheque, etc.).
- Real-time preview of cropped and scanned documents.
- Downloadable **JSON / CSV outputs** for extracted data.

---

## Tech Stack

| Component | Library/Tool |
|------------|--------------|
| UI | Streamlit |
| Document Cropping | streamlit-cropper, document-scanner-sdk |
| OCR | pytesseract |
| PDF Handling | pdfplumber, camelot |
| Vision Model | LLaVA (Ollama), Qwen2-VL |
| Backend | Python 3.10+ |
| JSON Parsing | regex + safe_eval logic |

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/document-intelligence-app.git
cd document-intelligence-app
```

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # for macOS/Linux
venv\Scripts\activate      # for Windows
```

### Install Requirements
```bash
pip install -r requirements.txt
```

### Install External Dependencies
#### Poppler (for PDF image conversion)
- **Windows:** [Download Poppler](http://blog.alivate.com.au/poppler-windows/)
- **macOS:** `brew install poppler`
- **Linux:** `sudo apt install poppler-utils`

#### Tesseract OCR
- **Windows:** [Download Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS/Linux:** `sudo apt install tesseract-ocr`

---

## Run the App

```bash
streamlit run app.py
```

Visit **http://localhost:8501** to use the app.

---

## How It Works

1. Upload an image or PDF.
2. Choose **document type** (Invoice, Cheque, etc.).
3. If image → app automatically crops document using **document-scanner-sdk**.
4. Vision LLM (Ollama / Qwen2-VL) extracts structured JSON data.
5. If model fails → fallback to OCR or pdfplumber.
6. Final structured JSON is displayed and can be **downloaded**.

---

## Document Types and Prompts

| Type | Example Use | Extraction Focus |
|------|--------------|------------------|
| Invoice / Bill | Nepa Wholesale invoices | Product, rate, amount, total |
| Bank Cheque | Handwritten cheques | Account no, name, amount |
| ID Document | Citizenship, Passport | Name, DOB, ID number |
| Generic Form | Utility forms | Field-value pairs |

---

## Example JSON Output

```json
{
  "product_code": "818036120292",
  "description": "POD JUICE X RAZ TOBACCO FREE NICOTINE E-JUICE 100ML",
  "quantity": 1,
  "rate": 6.75,
  "amount": 6.75
}
```

---

## Folder Structure

```
document-intelligence-app/
├── app.py                      # Main Streamlit app
├── utils/
│   ├── image_utils.py          # Cropping, preprocessing, OCR helpers
│   ├── pdf_utils.py            # Table extraction (pdfplumber, camelot)
│   ├── llm_utils.py            # Vision LLM extraction and prompt logic
├── requirements.txt
├── README.md
└── assets/
    └── sample_docs/            # Example images/PDFs
```

---

## Requirements

- Python ≥ 3.10  
- 8GB+ RAM (for Vision LLM inference)  
- If using **Ollama / LLaVA**: Ollama must be running locally  
  ```bash
  ollama serve
  ollama run llava
  ```

---

## Future Enhancements
- Add multi-language OCR (Nepali + English)
- Enable offline caching of models
- Integrate fine-tuned Qwen-VL for invoices
- Web API for external app integration
- Add RAG (Retrieval-Augmented Generation) for document QA

---

## License
MIT License © 2025 Aayush Adhikari
