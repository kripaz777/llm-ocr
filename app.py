import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import re
import json
import base64
from typing import List, Tuple, Dict, Any

from PIL import Image, UnidentifiedImageError
import cv2

# --- Optional: Super-resolution
from super_image import EdsrModel, ImageLoader

# --- Optional cropper
from streamlit_cropper import st_cropper

# --- Vision LLM (local) - Ollama
import ollama

# --- PDF -> Images
from pdf2image import convert_from_bytes

# --- OCR (no Tesseract)
import easyocr

# --- Optional vendor: if installed, use it automatically
try:
    from document_scanner_sdk import DocumentScanner
    USE_DOCSCAN = True
except ImportError:
    USE_DOCSCAN = False

# =========================
#        CONFIG
# =========================
st.set_page_config(page_title="Vision LLM OCR • PDF to Images", layout="wide")

# Vision-LLaMA model name
OLLAMA_VISION_MODEL = "qwen2.5vl:7b"

# Super-resolution model
SUPER_IMAGE_MODEL = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=2)

# Init OCR
@st.cache_resource(show_spinner=False)
def _init_easyocr_reader(langs=('en',)):
    return easyocr.Reader(list(langs), gpu=True)

# =========================
#     IMAGE UTILITIES
# =========================
def pil_from_upload(file) -> Image.Image | None:
    try:
        img = Image.open(file).convert("RGB")
        return img
    except UnidentifiedImageError:
        st.error(f"❌ Cannot open image: {getattr(file, 'name', 'image')}")
        return None

def resize_long_side(pil_img: Image.Image, max_side=1600) -> Image.Image:
    if max(pil_img.size) <= max_side:
        return pil_img
    pil_img = pil_img.copy()
    pil_img.thumbnail((max_side, max_side))
    return pil_img

def to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def to_rgb_img(arr_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))

def auto_crop(pil_img: Image.Image) -> Image.Image:
    try:
        img_cv = to_bgr(pil_img)

        if USE_DOCSCAN:
            scanner = DocumentScanner()
            scanned = scanner.scan(img_cv)
            return to_rgb_img(scanned)

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(gray, 60, 180)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:7]

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype("float32")

                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                tl = pts[np.argmin(s)]
                br = pts[np.argmax(s)]
                tr = pts[np.argmin(diff)]
                bl = pts[np.argmax(diff)]
                rect = np.array([tl, tr, br, bl], dtype="float32")

                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxW = int(max(widthA, widthB))

                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxH = int(max(heightA, heightB))

                dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img_cv, M, (maxW, maxH))
                return to_rgb_img(warped)

        return pil_img
    except Exception as e:
        st.warning(f"Auto-crop failed, using original image. Details: {e}")
        return pil_img

def maybe_super_res(pil_img: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return pil_img
    try:
        sr_in = ImageLoader.load_image(pil_img)
        sr_out = SUPER_IMAGE_MODEL(sr_in)
        return sr_out
    except Exception as e:
        st.warning(f"Upscale failed (continuing without it): {e}")
        return pil_img

# =========================
#         PDF UTIL
# =========================
def pdf_to_images(file_bytes: bytes, dpi=200) -> List[Image.Image]:
    images = convert_from_bytes(file_bytes, dpi=dpi, fmt="jpeg")
    images = [img.convert("RGB") for img in images]
    return images

# =========================
#         OCR UTIL
# =========================
def easy_ocr_text(pil_img: Image.Image, reader) -> str:
    img_np = np.array(pil_img)
    results = reader.readtext(img_np)
    print("\n".join([r[1] for r in results]) if results else "")
    return "\n".join([r[1] for r in results]) if results else ""

# =========================
#  OLLAMA UTIL
# =========================
def pil_to_base64_str(pil_img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def force_json_extract(text: str) -> Any | None:
    candidates = re.findall(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    for c in candidates[::-1]:
        try:
            return json.loads(c)
        except Exception:
            continue
    try:
        return json.loads(text)
    except Exception:
        return None

# =========================
#   PROMPT LOGIC
# =========================
def build_prompt_for_doc(doc_type: str, fields: List[str], instr: str, text: str | None = None) -> str:
    base = f"Extract the following fields as strict JSON.\nFields: {json.dumps(fields)}\nInstructions: {instr}\n"

    if doc_type == "Invoice":
        extra = "This is an invoice. Focus on table. Extract all data that the invoice has."
    elif doc_type == "Bill":
        extra = "This is a bill. Extract all data that have been mentioned."
    elif doc_type == "Bank Cheque":
        extra = """This is a bank cheque. Extract payee name, amount in words, amount in numbers, date, cheque number, 
        and account number.Amount might me in hand written.Extract those carefully."""
    else:
        extra = "This is a generic document. Extract text into the requested JSON fields."

    prompt = base + "\nContext: " + extra
    if text:
        prompt += f"\n\nOCR TEXT:\n{text}"

    return prompt

# =========================
#   EXTRACTION METHODS
# =========================
def llava_extract_with_image(pil_img: Image.Image, fields: List[str], user_instr: str, doc_type: str) -> Dict[str, Any] | None:
    img_b64 = pil_to_base64_str(pil_img, fmt="JPEG")

    prompt = build_prompt_for_doc(doc_type, fields, user_instr)

    try:
        resp = ollama.chat(
            model=OLLAMA_VISION_MODEL,
            messages=[
                {"role": "system",
                  "content": "You are an extraction engine. Extract the necessary information. Return a Json output with list of all items extracted from the imag"},
                {"role": "user",
                "content": prompt,
                "images": [img_b64]
                },
            ],
            options={"temperature": 1}
        )
        content = resp["message"]["content"]
        data = force_json_extract(content)

        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        if isinstance(data, dict):
            return data
        return None
    except Exception as e:
        st.warning(f"Vision LLM extraction failed: {e}")
        return None

def hybrid_extract(pil_img: Image.Image, fields: List[str], instr: str, reader, doc_type: str) -> Dict[str, Any] | None:
    text = easy_ocr_text(pil_img, reader)
    prompt = build_prompt_for_doc(doc_type, fields, instr, text)
    try:
        resp = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 1}
        )
        content = resp["message"]["content"]
        data = force_json_extract(content)
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            return data[0]
        if isinstance(data, dict):
            return data
        return None
    except Exception as e:
        st.warning(f"Hybrid (OCR+LLM) failed: {e}")
        return None

def ensure_json_result(record, pil_img, fields, reader, mode):
    if isinstance(record, dict):
        return record
    if isinstance(record, list) and len(record) > 0 and isinstance(record[0], dict):
        return record[0]

    text_fallback = easy_ocr_text(pil_img, reader) if mode != "Vision LLM (Ollama Qwen)" else ""
    fallback = {f: "" for f in fields}
    fallback["__raw_text"] = text_fallback
    fallback["__status"] = "LLM parse failed"
    return fallback

# =========================
#           UI
# =========================
st.title("Vision LLM OCR & PDF Extractor")
st.caption("Runs locally with Ollama (LLaVA) + EasyOCR fallback. Multi-page PDFs are converted to images first.")

with st.sidebar:
    st.header("Settings")
    doc_type = st.selectbox("Document Type", ["Invoice", "Bill", "Bank Cheque", "Other"], index=0)
    mode = st.selectbox("Extraction Mode", ["Vision LLM (Ollama LLaVA)", "OCR + LLM (hybrid)", "OCR only"], index=0)
    fields_input = st.text_input("Fields to extract (comma-separated)", "Invoice Number, Date, Vendor, Amount, Items")
    fields = [f.strip() for f in fields_input.split(",") if f.strip()]
    instructions = st.text_area("Optional instructions", "Return numbers as raw, dates as YYYY-MM-DD.")
    langs = st.multiselect("OCR languages", ["en", "de", "fr", "es"], default=["en"])
    upscale = st.checkbox("Upscale before OCR/LLM (EDSR ×2)", value=False)
    enable_auto_crop = st.checkbox("Auto-crop (document edge detect)", value=False)
    enable_manual_crop = st.checkbox("Manual crop after auto-crop", value=False)
    dpi = st.slider("PDF render DPI", 150, 300, 200, step=25)

uploaded = st.file_uploader("Upload PDFs or Images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

# Session storage
if "confirmed" not in st.session_state:
    st.session_state.confirmed = []

# =========================
#     FILE PREVIEW
# =========================
if uploaded:
    st.subheader("1) Review & Confirm Pages")
    reader = _init_easyocr_reader(tuple(langs))

    for file in uploaded:
        name = getattr(file, "name", "document")
        ext = os.path.splitext(name)[1].lower()

        with st.expander(f"📄 {name}", expanded=False):
            if ext == ".pdf":
                file_bytes = file.read()
                pages = pdf_to_images(file_bytes, dpi=dpi)
                st.info(f"Detected {len(pages)} page(s)")

                for idx, p in enumerate(pages, start=1):
                    base = resize_long_side(p, max_side=1600)
                    auto = auto_crop(base) if enable_auto_crop else base

                    col1, col2 = st.columns([1,1])
                    with col1: st.image(base, caption=f"Page {idx}: Original", use_container_width=True)
                    with col2: st.image(auto, caption=f"Page {idx}: Auto-cropped", use_container_width=True)

                    final_img = auto
                    if enable_manual_crop:
                        final_img = st_cropper(auto, realtime_update=False, box_color="#FF5A5F", return_type="image", key=f"crop_{name}_{idx}")

                    final_img = maybe_super_res(final_img, upscale)

                    if st.button(f"✅ Confirm Page {idx}", key=f"confirm_{name}_{idx}"):
                        st.session_state.confirmed.append((name, idx, final_img))
                        st.success(f"Added {name} p.{idx} to queue")

            else:
                img = pil_from_upload(file)
                if img is None:
                    continue
                base = resize_long_side(img, max_side=1600)
                auto = auto_crop(base) if enable_auto_crop else base
                final_img = auto
                if enable_manual_crop:
                    final_img = st_cropper(auto, realtime_update=False, box_color="#FF5A5F", return_type="image", key=f"crop_{name}")

                final_img = maybe_super_res(final_img, upscale)

                if st.button(f"✅ Confirm {name}", key=f"confirm_{name}"):
                    st.session_state.confirmed.append((name, 1, final_img))
                    st.success(f"Added {name} to queue")

# =========================
#     EXTRACTION RUN
# =========================
st.subheader("2) Run Extraction")
run = st.button("▶️ Run on Confirmed Items")
results: List[pd.DataFrame] = []
errors: List[Tuple[str, str]] = []

if run:
    if not st.session_state.get("confirmed"):
        st.warning("No confirmed pages yet.")
    else:
        with st.spinner("Extracting…"):
            reader = _init_easyocr_reader(tuple(langs))
            for (src, page, img) in st.session_state.confirmed:
                try:
                    record = None
                    if mode == "Vision LLM (Ollama Qwen)":
                        record = llava_extract_with_image(img, fields, instructions, doc_type)
                    elif mode == "OCR + LLM (hybrid)":
                        record = hybrid_extract(img, fields, instructions, reader, doc_type)
                    elif mode == "OCR only":
                        text = easy_ocr_text(img, reader)
                        record = {f: "" for f in fields}
                        record["__raw_text"] = text

                    record = ensure_json_result(record, img, fields, reader, mode)

                    df = pd.DataFrame([record])
                    df["Source_File"] = src
                    df["Page"] = page
                    df["Doc_Type"] = doc_type
                    results.append(df)

                except Exception as e:
                    errors.append((f"{src} (p.{page})", str(e)))

        if results:
            final = pd.concat(results, ignore_index=True)
            st.success(f"Extraction complete. {len(final)} row(s).")
            st.dataframe(final, use_container_width=True)
            st.session_state["last_df"] = final

            csv = final.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "extracted.csv", "text/csv")

        if errors:
            st.error("Some pages failed:")
            for src_page, msg in errors:
                st.write(f"• {src_page}: {msg}")

# =========================
# 3) Optional Re-run
# =========================
if st.session_state.get("last_df") is not None:
    st.subheader("3) Adjust & Re-run (optional)")
    new_instr = st.text_area("Refine instructions", value=instructions, key="refine_instr")
    if st.button("🔁 Re-run with new instructions"):
        st.session_state.pop("last_df")
        st.experimental_rerun()
