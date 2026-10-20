# Handwritten Medical Document OCR: Safe Fallback Pipeline

Part of an AI-assisted prior authorization and fraud risk project (Team Web).

When a scan is blurry or handwritten, the pipeline never guesses silently. Each word gets a
calibrated confidence score and is routed: auto-accept, human confirms, or human review (blocked
from the claim). A bad page triggers a request for a better copy.

## Pipeline
1. Image quality check (sharpness, contrast, size, exposure)
2. Preprocessing (stamp removal, background correction, denoise, sharpen, deskew, CLAHE, Sauvola binarisation)
3. OCR with per-word confidence (Tesseract baseline; production design: TrOCR / PaddleOCR / docTR)
4. PII masking (Aadhaar, phone) on text and image
5. Confidence calibration (logistic / Platt) and lexicon validation
6. Routing by calibrated confidence

## Run
```
pip install -r requirements.txt
sudo apt install tesseract-ocr        # or install Tesseract for your OS
python run_demo.py                     # synthetic blurry prescription
python run_demo.py --image my_scan.jpg # your own scan
```
Outputs (figures, metrics.json) are written to `outputs/`.

## Important notes
- All data is synthetic. No real patient data is included. Never commit real patient records.
- Handwriting is imitated with jittered letters; results are a method demonstration, not a final accuracy claim.
- Tesseract is a weak baseline for handwriting. Replace `run_ocr` in `pipeline.py` with TrOCR or PaddleOCR for real use.

## Files
- `synth.py` synthetic document and degradation generator
- `pipeline.py` quality gate, preprocessing, PII masking, OCR, calibration, routing, metrics
- `run_demo.py` runs everything and saves figures

## References
See the project documentation for the full list (TrOCR arXiv:2109.10282, PP-OCR arXiv:2009.09941,
Guo et al. calibration arXiv:1706.04599, Geifman and El-Yaniv selective classification arXiv:1705.08500).
