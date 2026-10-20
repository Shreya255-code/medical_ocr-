"""Handwritten / bad-scan OCR pipeline with calibrated confidence and safe routing.
"""
import re
import difflib
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from skimage.filters import threshold_sauvola
from sklearn.linear_model import LogisticRegression

#  1. QUALITY GATE
def quality_gate(img_bgr, min_blur=4.0, min_contrast=25.0, min_w=700):
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(gray, (1000, int(gray.shape[0] * 1000 / gray.shape[1])))  # scale-normalise
    blur = float(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_64F).var())  # low = blurry
    contrast = float(np.percentile(g, 95) - np.percentile(g, 5))
    brightness = float(g.mean())
    reasons = []
    if blur < min_blur: reasons.append(f"too blurry (sharpness {blur:.1f} < {min_blur:.1f})")
    if contrast < min_contrast: reasons.append(f"low contrast ({contrast:.0f} < {min_contrast:.0f})")
    if gray.shape[1] < min_w: reasons.append(f"low resolution (width {gray.shape[1]}px)")
    if brightness < 60 or brightness > 250: reasons.append(f"bad exposure (mean {brightness:.0f})")
    return {"sharpness": blur, "contrast": contrast, "brightness": brightness,
            "passed": not reasons, "reasons": reasons}

# 2. PREPROCESS
def remove_stamp(img_bgr):
    """Remove saturated red/purple stamp ink (inpaint). Dark pen ink is untouched."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = (((h < 12) | (h > 160)) & (s > 70) & (v > 60)).astype(np.uint8) * 255
    red = cv2.dilate(red, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.inpaint(img_bgr, red, 3, cv2.INPAINT_TELEA), red


def normalize_background(gray, paper_level=245):
    """Divide by estimated paper illumination -> removes shadows / uneven light."""
    k = max(31, (min(gray.shape) // 12) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    out = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * paper_level
    return out.clip(0, 255).astype(np.uint8)


def denoise(gray):
    return cv2.fastNlMeansDenoising(gray, None, h=9, templateWindowSize=7, searchWindowSize=21)


def sharpen(gray, amount=1.2, sigma=1.6):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def estimate_skew(gray, max_angle=15.0):
    """Projection-profile search: the angle whose row sums are most 'peaky'."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ink = (small < threshold_sauvola(small, 25, 0.2)).astype(np.float32)
    h, w = ink.shape

    def score(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR)
        return np.var(r.sum(axis=1))
    coarse = max(np.arange(-max_angle, max_angle + 0.01, 1.0), key=score)
    fine = max(np.arange(coarse - 1, coarse + 1.01, 0.1), key=score)
    return float(fine)


def rotate(img, angle, border=None):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    b = border if border is not None else (int(np.median(img)) if img.ndim == 2 else (200, 205, 210))
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=b)


def restore_photometric(img_bgr):
    """Geometry-preserving restoration. Returns dict of stage -> gray image (for MSE/MAE)."""
    st = {}
    nostamp, _ = remove_stamp(img_bgr)
    g0 = cv2.cvtColor(nostamp, cv2.COLOR_BGR2GRAY)
    st["1_stamp_removed"] = g0
    st["2_background_normalised"] = normalize_background(g0)
    st["3_denoised"] = denoise(st["2_background_normalised"])
    st["4_sharpened"] = sharpen(st["3_denoised"])
    return st


def preprocess(img_bgr, upscale=2.0):
    """Full pipeline for OCR. Returns (final_binary, stages dict, estimated skew angle)."""
    st = restore_photometric(img_bgr)
    gray = st["4_sharpened"]
    angle = estimate_skew(gray)
    st["5_deskewed"] = rotate(gray, angle)
    big = cv2.resize(st["5_deskewed"], None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    big = clahe.apply(big)
    st["6_contrast_clahe_upscaled"] = big
    binary = ((big > threshold_sauvola(big, 41, 0.18)) * 255).astype(np.uint8)
    st["7_binarised_sauvola"] = binary
    return binary, st, angle

#3. PII MASKING
AADHAAR = re.compile(r"^\d{4}$")
PHONE = re.compile(r"^[6-9]\d{9}$")

def _digit_heavy(t):
    d = sum(c.isdigit() for c in t)
    return len(t) >= 4 and d / len(t) >= 0.6


def _fuzzy_in(token, names, thr=0.6):
    t = re.sub(r"[^a-z]", "", token.lower())
    return len(t) >= 2 and any(difflib.SequenceMatcher(None, t, n).ratio() >= thr for n in names)


def mask_pii(words):
    """Redact Aadhaar and phone numbers BEFORE any text leaves the OCR stage.
    Handwriting makes OCR split/misread digits, so strict regex alone leaks PII. Three rules:
      1. exact patterns: 10-digit mobile, three consecutive 4-digit groups (Aadhaar)
      2. label rule: on a line containing 'Aadhaar/UID', redact every digit-heavy token
      3. label rule: a digit-heavy token right after 'Ph/Phone/Mobile' is redacted"""
    n, hit = len(words), set()
    for i, w in enumerate(words):
        if PHONE.match(w["text"]): hit.add(i)
        if i + 2 < n and all(AADHAAR.match(words[j]["text"]) for j in (i, i + 1, i + 2)):
            hit.update({i, i + 1, i + 2})
    lines = {}
    for i, w in enumerate(words): lines.setdefault(w["line"], []).append(i)
    for idx in lines.values():
        if any(_fuzzy_in(words[i]["text"], ["aadhaar", "aadhar", "uid"], 0.62) for i in idx):
            hit.update(i for i in idx if _digit_heavy(words[i]["text"]))
    for i in range(1, n):
        if _fuzzy_in(words[i - 1]["text"], ["ph", "phone", "mobile", "mob"], 0.99) and _digit_heavy(words[i]["text"]):
            hit.add(i)
    for i in hit:
        words[i]["text"] = "[REDACTED]"; words[i]["pii"] = True
    return words

#  4. OCR
def run_ocr(img, psm=6):
    d = pytesseract.image_to_data(img, config=f"--oem 1 --psm {psm}", output_type=Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) >= 0:
            words.append({"text": t, "orig": t, "raw_conf": float(d["conf"][i]) / 100.0,
                          "box": (d["left"][i], d["top"][i], d["width"][i], d["height"][i]),
                          "line": (d["block_num"][i], d["par_num"][i], d["line_num"][i]), "pii": False})
    return words

# 5. DOMAIN VALIDATION
LEXICON = """patient age aadhaar ph date diagnosis procedure acute appendicitis laparoscopic appendectomy
tab inj rx paracetamol ceftriaxone pantoprazole admit discharge mg tds bd od iv amoxicillin metformin
atorvastatin ibuprofen diclofenac ondansetron cholecystectomy hernia fracture pneumonia diabetes
hypertension fever cough sepsis cesarean hysterectomy angioplasty ml gm stat sos bid qid""".split()

def lexicon_similarity(token):
    t = re.sub(r"[^a-z]", "", token.lower())
    if len(t) < 2: return 0.0, None
    best = max(LEXICON, key=lambda w: difflib.SequenceMatcher(None, t, w).ratio())
    return difflib.SequenceMatcher(None, t, best).ratio(), best


def valid_date(token):
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", token)
    if not m: return None
    d, mo, y = map(int, m.groups())
    return 1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2100


def features(w):
    sim, best = lexicon_similarity(w["text"])
    dt = valid_date(w["text"])
    return [w["raw_conf"], sim, float(any(c.isdigit() for c in w["text"])),
            min(len(w["text"]), 15) / 15.0, 1.0 if dt else (-1.0 if dt is False else 0.0)], best

#  6. CALIBRATION
class Calibrator:
    """Platt-style logistic calibration: P(word correct | raw_conf, lexicon match, ...).
    Learned on a labelled validation split so that 0.60 really means ~60% correct."""
    def __init__(self):
        self.lr = LogisticRegression(C=2.0, max_iter=1000)
        self.fitted = False

    def fit(self, X, y):
        self.lr.fit(np.array(X), np.array(y)); self.fitted = True; return self

    def predict(self, X):
        return self.lr.predict_proba(np.array(X))[:, 1]

# 7. ROUTING
def route(conf, critical):
    hi, lo = (0.90, 0.60) if critical else (0.80, 0.50)
    if conf >= hi: return "AUTO"
    if conf >= lo: return "SUGGEST_CONFIRM"
    return "HUMAN_REVIEW"


def process_document(img_bgr, calibrator, use_preprocess=True, rfi_human_fraction=0.40):
    """Whole pipeline -> structured result."""
    gate = quality_gate(img_bgr)
    if use_preprocess:
        img, stages, angle = preprocess(img_bgr)
    else:
        img, stages, angle = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), {}, 0.0
    words = mask_pii(run_ocr(img))
    feats, bests = zip(*[features(w) for w in words]) if words else ([], [])
    probs = calibrator.predict(feats) if len(words) else []
    for w, f, b, p in zip(words, feats, bests, probs):
        w["features"], w["calibrated_conf"] = f, float(p)
        w["critical"] = bool(f[2] > 0 or f[1] >= 0.8)
        w["suggestion"] = b if (f[1] >= 0.8 and b and b != w["text"].lower()) else None
        w["action"] = "REDACTED" if w["pii"] else route(float(p), w["critical"])
    img = img.copy()
    for w in words:                                    # visual redaction: black-box PII on the stored image
        if w["pii"]:
            x, y, ww, hh = w["box"]; cv2.rectangle(img, (x - 4, y - 4), (x + ww + 4, y + hh + 4), 0, -1)
    n_human = sum(w["action"] == "HUMAN_REVIEW" for w in words)
    rfi = (not gate["passed"]) or (words and n_human / len(words) > rfi_human_fraction)
    return {"gate": gate, "angle": angle, "words": words, "stages": stages, "image": img, "rfi": bool(rfi),
            "downstream_text": " ".join(w["text"] for w in words if w["action"] == "AUTO"),
            "pending_confirmation_text": " ".join(w["text"] for w in words if w["action"] == "SUGGEST_CONFIRM")}

#  8. METRICS
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(ref_words, hyp_words):
    return edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)


def label_words(words, gt_text):
    """Mark each OCR word correct/incorrect by aligning it to ground-truth tokens."""
    gt = [t.lower() for t in gt_text.split()]
    hyp = [w.get("orig", w["text"]).lower() for w in words]
    sm = difflib.SequenceMatcher(None, hyp, gt, autojunk=False)
    ok = np.zeros(len(hyp), dtype=int)
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag == "equal": ok[i1:i2] = 1
    return ok


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any(): tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def mse(a, b):   # images scaled to 0..1
    return float(np.mean((a.astype(np.float32) / 255 - b.astype(np.float32) / 255) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) / 255 - b.astype(np.float32) / 255)))
