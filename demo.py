"""
Guitar Chord Classifier — Live Demo (Enhanced)

Usage:
  python demo.py                        # webcam (default, camera 0)
  python demo.py --source guitarr.mp4   # video file
  python demo.py --source guitarr.mp4 --save  # save annotated output
  python demo.py --no-roboflow          # skip fretboard crop (full frame)
  python demo.py --screen               # screen capture mode
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

# ─── Config ──────────────────────────────────────────────────────────────────

WEIGHTS_PATH   = "chord_classifier_weights_final_best.pth"
CLASS_NAMES    = ["C", "D", "Em", "F", "G"]
ROBOFLOW_KEY   = "[ENCRYPTION_KEY]"
WORKSPACE      = "test-kpcsq"
WORKFLOW_ID    = "find-fretboards"
ROBOFLOW_URL   = ""  # http://localhost:9001 or https://serverless.roboflow.com
FRET_PAD       = 0.15
INFER_EVERY_N  = 1
ROBOFLOW_EVERY = 6
DETECT_MAX_W   = 640

# ─── Smoothing config ────────────────────────────────────────────────────────

EMA_ALPHA            = 0.25   # lower = smoother, less reactive
BOX_SMOOTH_BETA      = 0.7
BAR_LERP_SPEED       = 0.25
TEMPERATURE          = 1.0    # keep at 1.0 — matches training

# ─── Stabilisation config ─────────────────────────────────────────────────────

# How many consecutive frames must agree before a chord is committed.
# Higher = more stable, slightly more latency.
CHORD_COMMIT_FRAMES  = 4

# Global fallback confidence threshold (used if chord not in per-class dict).
CONFIDENCE_THRESHOLD = 0.50

# Per-class thresholds — Em needs a higher bar because it looks like C.
# Tune these if specific chords are still mis-firing.
PER_CLASS_THRESHOLD  = {
    "C":  0.45,
    "D":  0.40,
    "Em": 0.45,
    "F":  0.40,
    "G":  0.40,
}

# Minimum probability gap between 1st and 2nd place before committing.
# Prevents flipping between two similarly-scored chords (e.g. C vs Em).
MIN_MARGIN           = 0.10

# Hysteresis: once a chord is committed, don't switch to a new one unless
# the new chord's smoothed probability exceeds the current chord's by this margin.
# This is the primary fix for the Em↔C oscillation.
HYSTERESIS_MARGIN    = 0.15

# Entropy threshold — if probability mass is too spread, output "?".
MAX_NORM_ENTROPY     = 0.82

# ─── OOD (no-chord) detection  [DISABLED — commented out for testing] ─────────
# IDLE_FRAMES_DIR    = "idle_frames"
# IDLE_SIM_THRESHOLD = 0.95

# ─── Visual config ────────────────────────────────────────────────────────────

PANEL_WIDTH       = 240
CHORD_HISTORY_LEN = 14

CHORD_COLOURS = {
    "C":  (0,   200, 255),
    "D":  (0,   255, 120),
    "Em": (255, 100,   0),
    "F":  (80,    0, 255),
    "G":  (200,   0, 255),
    "?":  (100, 100, 100),
}

CHORD_FINGERINGS = {
    "C":  [None, 3, 2, 0, 1, 0],
    "D":  [None, None, 0, 2, 3, 2],
    "Em": [0, 2, 2, 0, 0, 0],
    "F":  [1, 3, 3, 2, 1, 1],
    "G":  [3, 2, 0, 0, 0, 3],
}

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_TRIPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class GuitarChordClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.base_model = timm.create_model(
            "inception_resnet_v2", pretrained=False, num_classes=0
        )
        in_features = self.base_model.num_features
        self.custom_tail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.base_model.forward_features(x)
        return self.custom_tail(features)


def load_model(weights_path: str, device: torch.device):
    """Load weights into model. Returns (compiled_model, raw_model).
    raw_model is used for OOD feature extraction (torch.compile breaks
    direct attribute access on submodules)."""
    raw_model = GuitarChordClassifier(num_classes=len(CLASS_NAMES)).to(device)
    raw_model.load_state_dict(torch.load(weights_path, map_location=device))
    raw_model.eval()
    compiled_model = raw_model
    try:
        compiled_model = torch.compile(raw_model)
        print("[INFO] Model compiled with torch.compile()")
    except Exception as e:
        print(f"[INFO] torch.compile() skipped: {e}")
    return compiled_model, raw_model


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_crop(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR crop → sample-wise centered 229×229 tensor.

    Matches training exactly:
      transforms.Resize((229, 229))
      transforms.ToTensor()          → [0, 1]
      SampleWiseCenter()             → subtract image mean
    """
    rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (229, 229), interpolation=cv2.INTER_LINEAR)
    tensor  = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
    tensor  = tensor - tensor.mean()   # sample-wise centering
    return tensor.unsqueeze(0).to(device)


# ═══════════════════════════════════════════════════════════════════════════════
#  OOD DETECTION  (prototype-based, no retraining)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(crop_bgr: np.ndarray, raw_model: nn.Module,
                     device: torch.device) -> torch.Tensor:
    tensor = preprocess_crop(crop_bgr, device)
    with torch.inference_mode():
        feat_map = raw_model.base_model.forward_features(tensor)
        pooled   = raw_model.custom_tail[0](feat_map)
        vec      = raw_model.custom_tail[1](pooled)
    return torch.nn.functional.normalize(vec, dim=1)


def build_idle_prototype(idle_dir: str, raw_model: nn.Module,
                         device: torch.device):
    p = Path(idle_dir)
    images = (list(p.glob("*.jpg")) + list(p.glob("*.jpeg")) +
              list(p.glob("*.png"))) if p.exists() else []
    if not images:
        print(f"[INFO] No idle frames in '{idle_dir}' — OOD prototype disabled.")
        return None
    vecs = [extract_features(cv2.imread(str(img)), raw_model, device)
            for img in images if cv2.imread(str(img)) is not None]
    if not vecs:
        return None
    proto = torch.cat(vecs, dim=0).mean(dim=0, keepdim=True)
    print(f"[INFO] Idle prototype built from {len(vecs)} image(s).")
    return torch.nn.functional.normalize(proto, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  FUSED INFERENCE + OOD  (single backbone forward pass)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_and_check_ood(crop_bgr: np.ndarray, raw_model: nn.Module,
                           device: torch.device, idle_proto):
    """One backbone pass → (chord, confidence, probs_dict, is_ood).
    Costs the same as a single classify_crop() call."""
    tensor = preprocess_crop(crop_bgr, device)
    with torch.inference_mode():
        feat_map = raw_model.base_model.forward_features(tensor)
        pooled   = raw_model.custom_tail[0](feat_map)
        vec      = raw_model.custom_tail[1](pooled)

        # OOD detection disabled for testing
        ood = False

        x = vec
        for i in range(2, len(raw_model.custom_tail)):
            x = raw_model.custom_tail[i](x)
        probs = torch.softmax(x / TEMPERATURE, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return (CLASS_NAMES[idx], float(probs[idx]),
            {c: float(p) for c, p in zip(CLASS_NAMES, probs)}, ood)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION GATING  (the main fix for Em/C confusion)
# ═══════════════════════════════════════════════════════════════════════════════

def gate_prediction(smoothed_probs: dict, ood: bool,
                    committed_chord: str) -> tuple[str, float]:
    """Apply all stability gates and return (chord_or_question_mark, confidence).

    Gates applied in order:
      1. OOD prototype check  → "?"
      2. Entropy check        → "?"
      3. Per-class confidence → "?"
      4. Margin check         → "?"
      5. Hysteresis           → keep committed chord if new one isn't clearly better
    """
    if ood:
        return "?", 0.0

    probs_arr    = np.array(list(smoothed_probs.values()))
    norm_entropy = (-np.sum(probs_arr * np.log(probs_arr + 1e-9))
                    / np.log(len(probs_arr)))

    if norm_entropy > MAX_NORM_ENTROPY:
        return "?", float(probs_arr.max())

    # Sort classes by probability
    sorted_classes = sorted(smoothed_probs, key=smoothed_probs.get, reverse=True)
    best_cls       = sorted_classes[0]
    best_conf      = smoothed_probs[best_cls]
    second_conf    = smoothed_probs[sorted_classes[1]]

    # Per-class confidence threshold
    threshold = PER_CLASS_THRESHOLD.get(best_cls, CONFIDENCE_THRESHOLD)
    if best_conf < threshold:
        return "?", best_conf

    # Margin gate — winner must be clearly ahead of runner-up
    if (best_conf - second_conf) < MIN_MARGIN:
        return "?", best_conf

    # Hysteresis — if we already have a committed chord, the new one must
    # beat it by HYSTERESIS_MARGIN, not just be the argmax.
    # This is the key fix for Em ↔ C oscillation.
    if committed_chord not in ("?", ""):
        current_conf = smoothed_probs.get(committed_chord, 0.0)
        if best_cls != committed_chord:
            if (best_conf - current_conf) < HYSTERESIS_MARGIN:
                # Not confident enough to switch — stay on current chord
                return committed_chord, current_conf

    return best_cls, best_conf


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBOFLOW FRETBOARD DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_roboflow_client():
    try:
        from inference_sdk import InferenceHTTPClient
        return InferenceHTTPClient(api_url=ROBOFLOW_URL, api_key=ROBOFLOW_KEY)
    except ImportError:
        print("[WARN] inference_sdk not installed — falling back to full-frame.")
        return None


def detect_fretboard(client, frame_bgr: np.ndarray):
    try:
        import supervision as sv
        h, w = frame_bgr.shape[:2]
        if w > DETECT_MAX_W:
            scale = DETECT_MAX_W / w
            small = cv2.resize(frame_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            scale, small = 1.0, frame_bgr

        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE, workflow_id=WORKFLOW_ID,
                images={"image": small}, use_cache=True,
            )
        except Exception:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                tmp_path = f.name
            cv2.imwrite(tmp_path, small)
            result = client.run_workflow(
                workspace_name=WORKSPACE, workflow_id=WORKFLOW_ID,
                images={"image": tmp_path}, use_cache=True,
            )
            os.unlink(tmp_path)

        dets = sv.Detections.from_inference(result[0]["predictions"])
        if len(dets) == 0:
            return None
        best = int(np.argmax(dets.confidence)) if dets.confidence is not None else 0
        box  = dets.xyxy[best]
        if scale != 1.0:
            box = box / scale
        return tuple(map(int, box))
    except Exception as e:
        print(f"[WARN] Roboflow error: {e}")
        return None


def crop_with_padding(frame: np.ndarray, box, pad: float = FRET_PAD):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    px   = int((x2 - x1) * pad)
    py   = int((y2 - y1) * pad)
    return (frame[max(0, y1-py):min(h, y2+py),
                  max(0, x1-px):min(w, x2+px)],
            (max(0, x1-px), max(0, y1-py),
             min(w, x2+px), min(h, y2+py)))


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_box(new_box, prev_box, beta=BOX_SMOOTH_BETA):
    if prev_box is None:
        return new_box
    return tuple(int(beta * n + (1 - beta) * p)
                 for n, p in zip(new_box, prev_box))


RAW_PROB_BUFFER_LEN = 3  # average last N raw predictions before EMA

def smooth_probs(new_probs, prev_smoothed, raw_buffer, alpha=EMA_ALPHA):
    """Average last N raw predictions, then apply EMA on top."""
    raw_buffer.append(dict(new_probs))
    averaged = {c: np.mean([f[c] for f in raw_buffer]) for c in CLASS_NAMES}
    if prev_smoothed is None:
        return averaged
    return {c: alpha * averaged[c] + (1 - alpha) * prev_smoothed[c]
            for c in CLASS_NAMES}


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _gradient_panel(h: int, w: int) -> np.ndarray:
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        panel[y, :] = (int(20 + 22*t), int(12 + 16*t), int(12 + 18*t))
    return panel


def _glow_text(img, text, pos, font, scale, colour, thickness=2, glow_layers=3):
    x, y = pos
    dim  = tuple(max(0, min(255, int(c * 0.45))) for c in colour)
    for i in range(glow_layers, 0, -1):
        overlay = img.copy()
        cv2.putText(overlay, text, (x, y), font, scale + i*0.06,
                    dim, thickness + i*2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.12/i, img, 1 - 0.12/i, 0, img)
    cv2.putText(img, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def _confidence_ring(img, cx, cy, radius, confidence, colour, thick=3):
    cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, 360,
                (40, 40, 55), 2, cv2.LINE_AA)
    arc = int(360 * confidence)
    if arc > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, arc,
                    colour, thick, cv2.LINE_AA)
    pct = f"{int(confidence * 100)}%"
    sz  = cv2.getTextSize(pct, FONT_SMALL, 0.45, 1)[0]
    cv2.putText(img, pct, (cx - sz[0]//2, cy + sz[1]//2),
                FONT_SMALL, 0.45, (200, 200, 220), 1, cv2.LINE_AA)


def draw_fingering_diagram(panel, chord, y_top, pw):
    fingering = CHORD_FINGERINGS.get(chord)
    if fingering is None:
        return
    colour    = CHORD_COLOURS.get(chord, (180, 180, 200))
    n_strings, n_frets = 6, 4
    sg, fg, dot_r = 9, 11, 3
    total_w = (n_strings - 1) * sg
    ox      = (pw - total_w) // 2
    oy      = y_top + 16

    cv2.putText(panel, "FINGERING", (pw//2 - 27, y_top),
                FONT_SMALL, 0.30, (100, 100, 120), 1, cv2.LINE_AA)

    frets_used = [f for f in fingering if f is not None and f > 0]
    start_fret = 1
    if frets_used and max(frets_used) > n_frets:
        start_fret = min(frets_used)

    nut_thick = 3 if start_fret == 1 else 1
    cv2.line(panel, (ox, oy), (ox + total_w, oy), (200, 200, 215),
             nut_thick, cv2.LINE_AA)
    for j in range(1, n_frets + 1):
        cv2.line(panel, (ox, oy + j*fg), (ox + total_w, oy + j*fg),
                 (90, 90, 110), 1, cv2.LINE_AA)
    for i in range(n_strings):
        cv2.line(panel, (ox + i*sg, oy), (ox + i*sg, oy + n_frets*fg),
                 (90, 90, 110), 1, cv2.LINE_AA)
    if start_fret > 1:
        cv2.putText(panel, f"{start_fret}fr",
                    (ox + total_w + 4, oy + fg),
                    FONT_SMALL, 0.26, (150, 150, 170), 1, cv2.LINE_AA)

    for i, fret in enumerate(fingering):
        x = ox + i * sg
        if fret is None:
            cv2.line(panel, (x-3, oy-11), (x+3, oy-5), (160, 80, 80), 1, cv2.LINE_AA)
            cv2.line(panel, (x+3, oy-11), (x-3, oy-5), (160, 80, 80), 1, cv2.LINE_AA)
        elif fret == 0:
            cv2.circle(panel, (x, oy-8), 3, (100, 180, 100), 1, cv2.LINE_AA)
        else:
            rel = fret - start_fret + 1
            if 1 <= rel <= n_frets:
                cv2.circle(panel, (x, oy + (rel-1)*fg + fg//2),
                           dot_r, colour, -1, cv2.LINE_AA)


def draw_fretboard_box(frame, box, colour, confidence, fidx):
    x1, y1, x2, y2 = box
    pulse = 0.7 + 0.3 * math.sin(fidx * 0.15)
    glow  = tuple(int(c * pulse) for c in colour)
    roi   = frame[y1:y2, x1:x2]
    if roi.size > 0:
        tint = np.full_like(roi, colour, dtype=np.uint8)
        cv2.addWeighted(tint, 0.07, roi, 0.93, 0, roi)
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), glow, 2, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    tk, itk = 22, 14
    for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (sx, sy), (sx + dx*tk, sy), colour, 3, cv2.LINE_AA)
        cv2.line(frame, (sx, sy), (sx, sy + dy*tk), colour, 3, cv2.LINE_AA)
        cv2.line(frame, (sx+dx*4, sy+dy*4), (sx+dx*(4+itk), sy+dy*4),
                 glow, 1, cv2.LINE_AA)
        cv2.line(frame, (sx+dx*4, sy+dy*4), (sx+dx*4, sy+dy*(4+itk)),
                 glow, 1, cv2.LINE_AA)


def draw_title_bar(frame, status, fps, fidx):
    h, w  = frame.shape[:2]
    ov    = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 36), (15, 15, 25), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "GUITAR CHORD CLASSIFIER", (12, 25),
                FONT_SMALL, 0.48, (180, 180, 200), 1, cv2.LINE_AA)
    if int(fidx * 0.08) % 2 == 0:
        cv2.circle(frame, (w - 22, 18), 5, (0, 0, 220), -1, cv2.LINE_AA)
    fps_s = f"FPS: {fps:.0f}"
    fsz   = cv2.getTextSize(fps_s, FONT_SMALL, 0.38, 1)[0]
    cv2.putText(frame, fps_s, (w - 36 - fsz[0], 25),
                FONT_SMALL, 0.38, (140, 255, 140), 1, cv2.LINE_AA)
    cv2.putText(frame, status, (w - 36 - fsz[0] - 120, 25),
                FONT_SMALL, 0.38, (200, 200, 80), 1, cv2.LINE_AA)


def draw_chord_panel(frame, chord, confidence, all_probs, bar_display,
                     chord_history, trans_alpha, fidx, chord_start_time=None):
    h, w   = frame.shape[:2]
    pw     = PANEL_WIDTH
    panel  = _gradient_panel(h, pw)
    colour = CHORD_COLOURS.get(chord, (255, 255, 255))

    cv2.putText(panel, "DETECTED CHORD", (14, 30), FONT_SMALL, 0.40,
                (120, 120, 140), 1, cv2.LINE_AA)
    cv2.line(panel, (14, 38), (pw-14, 38), (50, 50, 65), 1, cv2.LINE_AA)

    disp  = chord if chord != "?" else "?"
    scale = 3.0 if len(disp) <= 2 else 2.2
    tsz   = cv2.getTextSize(disp, FONT_BOLD, scale, 3)[0]
    tx    = (pw - tsz[0]) // 2
    ty    = 108

    if trans_alpha > 0:
        flash = panel.copy()
        cv2.rectangle(flash, (0, 42), (pw, 125), colour, -1)
        cv2.addWeighted(flash, trans_alpha * 0.25, panel,
                        1 - trans_alpha * 0.25, 0, panel)

    _glow_text(panel, disp, (tx, ty), FONT_BOLD, scale, colour, 3, 3)

    if chord == "?":
        sub = "Detecting..."
        ssz = cv2.getTextSize(sub, FONT_SMALL, 0.36, 1)[0]
        cv2.putText(panel, sub, ((pw - ssz[0])//2, 128),
                    FONT_SMALL, 0.36, (140, 140, 155), 1, cv2.LINE_AA)
    elif chord_start_time is not None:
        hold_s = f"held  {time.time() - chord_start_time:.1f}s"
        hsz    = cv2.getTextSize(hold_s, FONT_SMALL, 0.33, 1)[0]
        cv2.putText(panel, hold_s, ((pw - hsz[0])//2, 130),
                    FONT_SMALL, 0.33, (140, 200, 140), 1, cv2.LINE_AA)

    _confidence_ring(panel, pw//2, 172, 28, confidence, colour)
    cv2.putText(panel, "CONFIDENCE", (pw//2 - 40, 212),
                FONT_SMALL, 0.33, (110, 110, 130), 1, cv2.LINE_AA)

    cv2.putText(panel, "PROBABILITIES", (14, 240),
                FONT_SMALL, 0.36, (120, 120, 140), 1, cv2.LINE_AA)
    cv2.line(panel, (14, 248), (pw-14, 248), (50, 50, 65), 1, cv2.LINE_AA)

    bar_top, bar_h, gap = 260, 20, 8
    bx1, bx2 = 14, pw - 14
    bw = bx2 - bx1

    for i, cls in enumerate(CLASS_NAMES):
        target = all_probs.get(cls, 0.0)
        cur    = bar_display.get(cls, 0.0)
        bar_display[cls] = cur + BAR_LERP_SPEED * (target - cur)
        p      = bar_display[cls]
        by1    = bar_top + i * (bar_h + gap)
        by2    = by1 + bar_h
        col    = CHORD_COLOURS.get(cls, (180, 180, 180))
        active = (cls == chord and chord != "?")

        cv2.rectangle(panel, (bx1, by1), (bx2, by2), (30, 30, 45), -1)
        fw = max(0, int(bw * p))
        if fw > 0:
            fc = col if active else tuple(int(c * 0.4) for c in col)
            cv2.rectangle(panel, (bx1, by1), (bx1 + fw, by2), fc, -1)
        if active:
            cv2.rectangle(panel, (bx1, by1), (bx2, by2), col, 1, cv2.LINE_AA)
        lbl = f"{cls}  {int(p * 100)}%"
        tc  = (255, 255, 255) if active else (150, 150, 160)
        cv2.putText(panel, lbl, (bx1+5, by2-5), FONT_SMALL, 0.36,
                    tc, 1, cv2.LINE_AA)

    hy = bar_top + len(CLASS_NAMES) * (bar_h + gap) + 18
    cv2.putText(panel, "HISTORY", (14, hy),
                FONT_SMALL, 0.33, (110, 110, 130), 1, cv2.LINE_AA)
    cv2.line(panel, (14, hy+6), (pw-14, hy+6), (50, 50, 65), 1, cv2.LINE_AA)

    dot_y, ds, dg = hy + 25, 14, 4
    for j, (ch, _) in enumerate(chord_history):
        dx = 14 + j * (ds + dg)
        if dx + ds > pw - 14:
            break
        c = CHORD_COLOURS.get(ch, (80, 80, 80))
        cv2.rectangle(panel, (dx, dot_y-ds//2), (dx+ds, dot_y+ds//2), c, -1)
        cv2.putText(panel, ch[0] if ch != "?" else "?",
                    (dx+2, dot_y+4), FONT_SMALL, 0.26,
                    (220, 220, 220), 1, cv2.LINE_AA)

    diag_y = dot_y + ds//2 + 18
    if chord != "?" and diag_y + 75 < h:
        cv2.line(panel, (14, diag_y-6), (pw-14, diag_y-6),
                 (50, 50, 65), 1, cv2.LINE_AA)
        draw_fingering_diagram(panel, chord, diag_y, pw)

    out = np.zeros((h, w + pw, 3), dtype=frame.dtype)
    out[:h, :w]       = frame
    out[:h, w:w + pw] = panel[:h]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  SCREEN CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

class ScreenCapture:
    def __init__(self, region=None):
        try:
            import mss
            self.sct = mss.mss()
        except ImportError:
            raise RuntimeError("pip install mss")
        self.monitor = ({"left": region[0], "top": region[1],
                         "width": region[2], "height": region[3]}
                        if region else self.sct.monitors[1])
        self._opened = True

    def isOpened(self): return self._opened

    def read(self):
        try:
            img   = np.array(self.sct.grab(self.monitor))
            return True, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:  return self.monitor["width"]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return self.monitor["height"]
        if prop == cv2.CAP_PROP_FPS:          return 30
        return 0

    def set(self, *_): pass
    def release(self): self._opened = False


def select_screen_region():
    import mss
    with mss.mss() as sct:
        mon  = sct.monitors[1]
        img  = np.array(sct.grab(mon))
        full = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w  = full.shape[:2]
    scale = min(1.0, 1920 / w)
    disp  = cv2.resize(full, None, fx=scale, fy=scale) if scale < 1 else full.copy()
    print("[INFO] Draw a rectangle around the capture area, then press ENTER.")
    roi = cv2.selectROI("Select Screen Region", disp, False, True)
    cv2.destroyWindow("Select Screen Region")
    if roi[2] == 0 or roi[3] == 0:
        return None
    x, y, rw, rh = roi
    return (int(x/scale)+mon["left"], int(y/scale)+mon["top"],
            int(rw/scale), int(rh/scale))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run(source, use_roboflow=True, save_output=False,
        output_path="output_demo.mp4", screen_mode=False):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading model from '{WEIGHTS_PATH}' …")
    model, raw_model = load_model(WEIGHTS_PATH, device)
    print("[INFO] Model ready.")

    # OOD detection disabled for testing
    idle_proto      = None
    roboflow_client = build_roboflow_client() if use_roboflow else None

    if screen_mode:
        region = select_screen_region()
        cap    = ScreenCapture(region=region)
        is_webcam = False
    else:
        is_webcam = (source is None or str(source) == "0")
        cap       = cv2.VideoCapture(0 if is_webcam else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    src_label = "screen" if screen_mode else ("webcam" if is_webcam else source)
    print(f"[INFO] Source: {src_label}  ({orig_w}×{orig_h} @ {fps_in:.1f} fps)")

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in,
                                 (orig_w + PANEL_WIDTH, orig_h))
        print(f"[INFO] Saving to '{output_path}'")

    # ── State ─────────────────────────────────────────────────────────────
    smooth_box_state = None
    last_chord       = "?"
    last_confidence  = 0.0
    smoothed_probs   = None
    raw_prob_buffer  = deque(maxlen=RAW_PROB_BUFFER_LEN)
    bar_display      = {c: 0.0 for c in CLASS_NAMES}
    chord_history    = deque(maxlen=CHORD_HISTORY_LEN)
    trans_alpha      = 0.0
    prev_chord       = "?"
    candidate_chord  = "?"
    chord_vote_count = 0
    chord_start_time = None
    frame_idx        = 0
    fps_display      = 0.0
    t_prev           = time.time()

    win = "Guitar Chord Classifier"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, orig_w + PANEL_WIDTH, orig_h)
    print("[INFO] Press  Q  to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if not is_webcam and not screen_mode:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        frame_idx += 1

        # ── Fretboard detection ────────────────────────────────────────────
        if roboflow_client and frame_idx % ROBOFLOW_EVERY == 1:
            raw_box = detect_fretboard(roboflow_client, frame)
            if raw_box:
                smooth_box_state = smooth_box(raw_box, smooth_box_state)

        active_box = smooth_box_state
        if active_box:
            crop, crop_box = crop_with_padding(frame, active_box)
        else:
            crop, crop_box = frame, (0, 0, orig_w, orig_h)

        # ── Inference ─────────────────────────────────────────────────────
        if frame_idx % INFER_EVERY_N == 0 and crop.size > 0:
            _, _, raw_probs, ood = classify_and_check_ood(
                crop, raw_model, device, idle_proto
            )
            smoothed_probs = smooth_probs(raw_probs, smoothed_probs, raw_prob_buffer)

            # ── Multi-gate prediction ────────────────────────────────────
            # Pass the currently-committed chord for hysteresis check.
            gated_chord, gated_conf = gate_prediction(
                smoothed_probs, ood, last_chord
            )

            last_confidence = gated_conf

            # ── Temporal vote gate ───────────────────────────────────────
            # Even after passing all probability gates, the chord must be
            # stable for CHORD_COMMIT_FRAMES consecutive frames.
            if gated_chord == candidate_chord:
                chord_vote_count += 1
            else:
                candidate_chord  = gated_chord
                chord_vote_count = 1

            if chord_vote_count >= CHORD_COMMIT_FRAMES:
                last_chord = candidate_chord

        # ── Chord-change transition ────────────────────────────────────────
        if last_chord != prev_chord:
            trans_alpha = 1.0
            if last_chord != "?":
                chord_history.append((last_chord, time.time()))
                chord_start_time = time.time()
            prev_chord = last_chord
        else:
            trans_alpha = max(0.0, trans_alpha - 0.06)

        # ── FPS ───────────────────────────────────────────────────────────
        t_now       = time.time()
        fps_display = 0.9 * fps_display + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev      = t_now

        # ── Draw ──────────────────────────────────────────────────────────
        chord_colour = CHORD_COLOURS.get(last_chord, (255, 255, 255))
        if active_box:
            draw_fretboard_box(frame, crop_box, chord_colour,
                               last_confidence, frame_idx)

        status = ("Screen" if screen_mode
                  else "Roboflow ON" if roboflow_client else "Full-frame")
        draw_title_bar(frame, status, fps_display, frame_idx)

        disp_probs = smoothed_probs or {c: 0.0 for c in CLASS_NAMES}
        out_frame  = draw_chord_panel(
            frame, last_chord, last_confidence, disp_probs,
            bar_display, chord_history, trans_alpha, frame_idx,
            chord_start_time,
        )

        if writer:
            writer.write(out_frame)

        cv2.imshow(win, out_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Guitar Chord Classifier — Enhanced Live Demo"
    )
    parser.add_argument("--source", type=str, default=None,
                        help="Video file path, or omit for webcam.")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output to file.")
    parser.add_argument("--output", type=str, default="output_demo.mp4")
    parser.add_argument("--no-roboflow", dest="no_roboflow",
                        action="store_true",
                        help="Skip fretboard detection; use full frame.")
    parser.add_argument("--screen", action="store_true",
                        help="Screen capture mode (requires: pip install mss).")
    args = parser.parse_args()

    run(
        source=args.source,
        use_roboflow=not args.no_roboflow,
        save_output=args.save,
        output_path=args.output,
        screen_mode=args.screen,
    )