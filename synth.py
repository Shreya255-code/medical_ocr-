"""Synthetic 'handwritten-style' prescription generator + phone-photo degradation.

IMPORTANT: this is a STAND-IN for real handwriting (no real patient data, no real
handwriting dataset is available offline). Each character is drawn separately with
random rotation, baseline wobble, scale and stroke jitter so it is irregular, like
handwriting. Replace with real consented, de-identified scans for your final results.
"""
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSerifItalic.ttf"
W, H = 1100, 760

# Fake patient data (all invented). Aadhaar/phone are here to demo PII masking.
LINES = [
    "Patient: Ramesh Kumar  Age 54  M",
    "Aadhaar 4821 5693 7710  Ph 9876543210",
    "Date 12/09/2026",
    "Diagnosis: Acute appendicitis",
    "Procedure: Laparoscopic appendectomy",
    "Rx Tab Paracetamol 500 mg TDS",
    "Inj Ceftriaxone 1 g IV BD",
    "Tab Pantoprazole 40 mg OD",
    "Admit 12/09/2026  Discharge 15/09/2026",
]


def render_clean(seed=0, jitter=0.8):
    # jitter = handwriting messiness dial. 1.0 = very messy (Tesseract gets ~5% of words),
    # 0.3 = neat handwriting (gives a realistic mix of right/wrong words so the confidence
    # and routing logic can be demonstrated). Real doctor handwriting needs TrOCR-class models.
    """Return (clean_gray uint8, ground_truth_text_lines)."""
    rng = random.Random(seed)
    img = Image.new("L", (W, H), 245)
    d = ImageDraw.Draw(img)
    # faint ruled lines like a prescription pad
    for y in range(110, H, 78):
        d.line([(40, y), (W - 40, y)], fill=225, width=1)
    y = 55
    for line in LINES:
        x = 50
        size = rng.randint(40, 46)
        font = ImageFont.truetype(FONT_PATH, size)
        for ch in line:
            if ch == " ":
                x += font.getlength(' ') * rng.uniform(0.9, 1.3)
                continue
            glyph = Image.new("L", (size * 2, size * 2), 0)
            gd = ImageDraw.Draw(glyph)
            gd.text((size * 0.4, size * 0.2), ch, font=font, fill=255)
            ang = rng.uniform(-9, 9) * jitter
            glyph = glyph.rotate(ang, resample=Image.BICUBIC)
            sc = rng.uniform(1 - 0.1 * jitter, 1 + 0.12 * jitter)
            glyph = glyph.resize((int(glyph.width * sc), int(glyph.height * sc)))
            dy = int(rng.uniform(-4, 4) * jitter)
            ink = rng.randint(15, 60)
            mask = np.array(glyph, dtype=np.float32) / 255.0
            canvas = np.array(img, dtype=np.float32)
            ys, xs = int(y + dy), int(x)
            hh, ww = mask.shape
            hh = min(hh, H - ys); ww = min(ww, W - xs)
            region = canvas[ys:ys + hh, xs:xs + ww]
            canvas[ys:ys + hh, xs:xs + ww] = region * (1 - mask[:hh, :ww]) + ink * mask[:hh, :ww]
            img = Image.fromarray(canvas.astype(np.uint8))
            x += font.getlength(ch) * rng.uniform(0.97, 1.08)
        y += 78
        d = ImageDraw.Draw(img)
    return np.array(img), LINES


def add_stamp(gray3):
    """Red circular hospital stamp, semi-transparent, over part of the text."""
    h, w = gray3.shape[:2]
    overlay = gray3.copy()
    cv2.circle(overlay, (int(w * 0.80), int(h * 0.40)), 95, (40, 40, 200), 6)
    cv2.circle(overlay, (int(w * 0.80), int(h * 0.40)), 70, (40, 40, 200), 3)
    cv2.putText(overlay, "CITY HOSPITAL", (int(w * 0.80) - 78, int(h * 0.40) + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 200), 2)
    return cv2.addWeighted(overlay, 0.75, gray3, 0.25, 0)


def photometric_degrade(clean_gray, severity, seed=0, with_stamp=True):
    """Blur + shadow + noise + stamp + low-res + JPEG. NO geometric change.
    Returns BGR uint8. severity in [0,1]."""
    rng = np.random.RandomState(seed)
    s = severity
    img = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
    # paper tint (phone photos are warm)
    img = (img.astype(np.float32) * np.array([0.93, 0.97, 1.0])).clip(0, 255).astype(np.uint8)
    if with_stamp:
        img = add_stamp(img)
    # defocus + motion blur
    sigma = 0.4 + 3.2 * s
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    k = int(1 + 2 * round(6 * s))
    if k > 1:
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        ang = rng.uniform(0, 180)
        M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), ang, 1)
        kern = cv2.warpAffine(kern, M, (k, k)); kern /= max(kern.sum(), 1e-6)
        img = cv2.filter2D(img, -1, kern)
    # uneven lighting / shadow
    h, w = img.shape[:2]
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    a = rng.uniform(0, 2 * np.pi)
    light = 1.0 - (0.15 + 0.45 * s) * (0.5 + 0.5 * np.cos(a) * (xx - .5) * 2 + 0.5 * np.sin(a) * (yy - .5) * 2).clip(0, 1)
    img = (img.astype(np.float32) * light[..., None]).clip(0, 255)
    # sensor noise
    img = img + rng.normal(0, 4 + 22 * s, img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    # low resolution (phone zoom / compression)
    f = 1.0 - 0.45 * s
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # JPEG artefacts
    q = int(92 - 55 * s)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def geometric_degrade(img_bgr, angle_deg, persp=0.0, seed=0):
    """Rotate (skew) + mild perspective, like a photo taken at an angle."""
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    out = cv2.warpAffine(img_bgr, M, (w, h), borderValue=(200, 205, 210))
    if persp > 0:
        rng = np.random.RandomState(seed)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-persp, persp, src.shape).astype(np.float32) * np.float32([w, h])
        P = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, P, (w, h), borderValue=(200, 205, 210))
    return out
