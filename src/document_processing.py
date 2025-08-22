#!/usr/bin/env python3
"""
Document Processing Pipeline

Reads raw financial documents from data/raw and produces a single consolidated
plain-text file in data/processed with light logical section segmentation.

Supported inputs (best-effort with graceful fallbacks):
- PDF: pypdf/PyPDF2, pdfminer.six, PyMuPDF (fitz); optional OCR via easyocr
- Images (png, jpg, jpeg, bmp, tiff): OCR via easyocr
- HTML/HTM: BeautifulSoup to extract text
- Excel (xls, xlsx): pandas to extract visible text
- DOCX: python-docx if available (optional)

Output:
- data/processed/consolidated_documents_YYYYMMDD_HHMMSS.txt
- data/processed/consolidated_documents_latest.txt (pointer copy)

Notes:
- OCR for scanned PDFs is attempted if text extraction is insufficient and
  PyMuPDF is available to render pages. This can be slow; we limit OCR pages.
"""

from __future__ import annotations

import os
import re
import sys
import io
import shutil
import datetime as _dt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Optional imports with graceful fallback
_HAVE_PYPDF = False
_HAVE_PDFMINER = False
_HAVE_FITZ = False
_HAVE_EASYOCR = False
_HAVE_BS4 = False
_HAVE_PANDAS = False
_HAVE_DOCX = False

try:
    import pypdf  # modern package name
    _HAVE_PYPDF = True
except Exception:
    try:
        from PyPDF2 import PdfReader  # legacy import
        _HAVE_PYPDF = True
    except Exception:
        _HAVE_PYPDF = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    _HAVE_PDFMINER = True
except Exception:
    _HAVE_PDFMINER = False

try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False

try:
    import easyocr
    _HAVE_EASYOCR = True
except Exception:
    _HAVE_EASYOCR = False

try:
    from bs4 import BeautifulSoup
    _HAVE_BS4 = True
except Exception:
    _HAVE_BS4 = False

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False

try:
    import docx  # python-docx
    _HAVE_DOCX = True
except Exception:
    _HAVE_DOCX = False


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\x0b\x0c]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path, max_ocr_pages: int = 3) -> str:
    """Best-effort text extraction from a PDF with graceful fallbacks.

    Order: pypdf/PyPDF2 -> pdfminer.six -> PyMuPDF text -> PyMuPDF+OCR (limited pages)
    """
    text_chunks: List[str] = []

    # Try pypdf/PyPDF2
    if _HAVE_PYPDF:
        try:
            if hasattr(pypdf, "PdfReader"):
                reader = pypdf.PdfReader(str(pdf_path))
                for page in reader.pages:
                    text = page.extract_text() or ""
                    if text:
                        text_chunks.append(text)
            else:
                # legacy PdfReader import path used above
                reader = PdfReader(str(pdf_path))  # type: ignore[name-defined]
                for page in reader.pages:
                    text = page.extract_text() or ""
                    if text:
                        text_chunks.append(text)
        except Exception:
            pass

    if not text_chunks and _HAVE_PDFMINER:
        try:
            text = pdfminer_extract_text(str(pdf_path)) or ""
            if text:
                text_chunks.append(text)
        except Exception:
            pass

    # PyMuPDF can sometimes extract text where pypdf/pdfminer struggle
    if not text_chunks and _HAVE_FITZ:
        try:
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    text = page.get_text("text") or ""
                    if text:
                        text_chunks.append(text)
        except Exception:
            pass

    # OCR fallback for scanned PDFs (limited pages)
    if not text_chunks and _HAVE_FITZ and _HAVE_EASYOCR:
        try:
            reader = easyocr.Reader(["en"], gpu=False)
            with fitz.open(str(pdf_path)) as doc:
                for pi, page in enumerate(doc):
                    if pi >= max_ocr_pages:
                        break
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    # easyocr expects path or numpy array; feed bytes via memory buffer
                    # Convert bytes to np array lazily via PIL to avoid hard dep here
                    try:
                        from PIL import Image
                        import numpy as np
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img_np = np.array(img)
                        ocr_results = reader.readtext(img_np, detail=0)
                        if ocr_results:
                            text_chunks.append("\n".join(ocr_results))
                    except Exception:
                        # Fallback: save temp file (last resort)
                        tmp = pdf_path.parent / f".__tmp_ocr_{pi}.png"
                        try:
                            with open(tmp, "wb") as f:
                                f.write(img_bytes)
                            ocr_results = reader.readtext(str(tmp), detail=0)
                            if ocr_results:
                                text_chunks.append("\n".join(ocr_results))
                        finally:
                            if tmp.exists():
                                tmp.unlink(missing_ok=True)
        except Exception:
            pass

    combined = "\n\n".join(text_chunks)
    return _normalize_whitespace(combined)


def extract_text_from_image(img_path: Path) -> str:
    if not _HAVE_EASYOCR:
        return ""
    try:
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(str(img_path), detail=0)
        return _normalize_whitespace("\n".join(results))
    except Exception:
        return ""


def extract_text_from_html(html_path: Path) -> str:
    if not _HAVE_BS4:
        return ""
    try:
        raw = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        return _normalize_whitespace(text)
    except Exception:
        return ""


def extract_text_from_excel(xl_path: Path) -> str:
    if not _HAVE_PANDAS:
        return ""
    try:
        xls = pd.ExcelFile(str(xl_path))
        parts: List[str] = []
        for sheet in xls.sheet_names:
            df = xls.parse(sheet, dtype=str, header=None)
            # Flatten cells; keep simple textual content
            values = df.fillna("").astype(str).values.tolist()
            lines = ["\t".join(row) for row in values if any(cell.strip() for cell in row)]
            if lines:
                parts.append(f"[Sheet: {sheet}]\n" + "\n".join(lines))
        return _normalize_whitespace("\n\n".join(parts))
    except Exception:
        return ""


def extract_text_from_docx(docx_path: Path) -> str:
    if not _HAVE_DOCX:
        return ""
    try:
        document = docx.Document(str(docx_path))
        lines = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        return _normalize_whitespace("\n".join(lines))
    except Exception:
        return ""


SECTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "INCOME_STATEMENT": [
        re.compile(r"\b(income\s+statement|statement\s+of\s+operations)\b", re.I),
    ],
    "BALANCE_SHEET": [
        re.compile(r"\b(balance\s+sheet|statement\s+of\s+financial\s+position)\b", re.I),
    ],
    "CASH_FLOW": [
        re.compile(r"\b(cash\s+flow|statement\s+of\s+cash\s+flows)\b", re.I),
    ],
    "MD&A": [
        re.compile(r"management'?s?\s+discussion\s+and\s+analysis", re.I),
        re.compile(r"md&a", re.I),
    ],
    "NOTES": [
        re.compile(r"\b(notes?\s+to\s+financial\s+statements?)\b", re.I),
    ],
}


def segment_financial_sections(text: str) -> List[Tuple[str, str]]:
    """Lightweight sectionizer: finds known headings and splits content.

    Returns a list of (section_name, section_text) covering the entire text,
    labeling unmatched spans as "UNCLASSIFIED".
    """
    if not text:
        return []

    # Find all heading matches with their positions
    matches: List[Tuple[int, str]] = []
    for label, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            for m in pat.finditer(text):
                matches.append((m.start(), label))

    if not matches:
        return [("UNCLASSIFIED", text)]

    matches.sort(key=lambda x: x[0])
    sections: List[Tuple[str, str]] = []

    # Build spans between headings
    for i, (start_idx, label) in enumerate(matches):
        end_idx = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        span_text = text[start_idx:end_idx].strip()
        if span_text:
            sections.append((label, span_text))

    # Prepend any content before the first heading as UNCLASSIFIED
    first_start = matches[0][0]
    if first_start > 0:
        preamble = text[:first_start].strip()
        if preamble:
            sections = [("UNCLASSIFIED", preamble)] + sections

    return sections


def process_file(path: Path) -> str:
    """Extract text from a single file with best-effort method."""
    suffix = path.suffix.lower()
    if suffix in {".pdf"}:
        return extract_text_from_pdf(path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return extract_text_from_image(path)
    if suffix in {".html", ".htm"}:
        return extract_text_from_html(path)
    if suffix in {".xls", ".xlsx"}:
        return extract_text_from_excel(path)
    if suffix in {".docx"}:
        return extract_text_from_docx(path)
    if suffix in {".txt"}:
        try:
            return _normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return ""

    # Unsupported types: return empty
    return ""


def consolidate_raw_documents(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    sectionize: bool = True
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in raw_dir.iterdir() if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files found in {raw_dir}")

    parts: List[str] = []
    for file_path in files:
        try:
            text = process_file(file_path)
            if not text:
                continue

            header = (
                f"===== FILE: {file_path.name} =====\n"
                f"[SOURCE_PATH]: {file_path.resolve()}\n"
                f"[EXTRACTED_AT]: {_dt.datetime.now().isoformat()}\n"
            )

            if sectionize:
                sections = segment_financial_sections(text)
                body_parts = []
                for label, content in sections:
                    body_parts.append(f"\n[Section: {label}]\n{content}\n")
                body = "".join(body_parts).strip()
            else:
                body = text

            parts.append(f"{header}\n{body}\n")
        except Exception as e:
            parts.append(
                f"===== FILE: {file_path.name} =====\n[ERROR] Failed to process: {e}\n"
            )

    if not parts:
        raise RuntimeError("No text could be extracted from raw documents.")

    stamp = _now_stamp()
    out_path = processed_dir / f"consolidated_documents_{stamp}.txt"
    out_path.write_text("\n\n".join(parts), encoding="utf-8")

    # Also write a latest pointer copy for convenience
    latest_path = processed_dir / "consolidated_documents_latest.txt"
    try:
        shutil.copyfile(out_path, latest_path)
    except Exception:
        # Non-fatal
        pass

    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Process raw financial documents into consolidated text")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR), help="Directory with raw documents")
    parser.add_argument("--out-dir", type=str, default=str(PROCESSED_DIR), help="Output directory for consolidated text")
    parser.add_argument("--no-sectionize", action="store_true", help="Disable financial section segmentation")
    args = parser.parse_args(argv)

    try:
        out = consolidate_raw_documents(
            raw_dir=Path(args.raw_dir),
            processed_dir=Path(args.out_dir),
            sectionize=(not args.no_sectionize),
        )
        print(f"[✓] Consolidated file saved at: {out}")
        return 0
    except Exception as e:
        print(f"[X] Document processing failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


