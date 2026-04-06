# Guitar Chord Classifier — Demo Implementation Details

The `demo.py` script serves as the live robust presentation layer for the Guitar Chord Classifier pipeline. It captures a video stream (webcam or file), locates the guitar fretboard in the frame, classifies the chord being played, and overlays a highly polished, responsive visual interface.

This document details the methodology, optimizations, and internal components that power the live demo.

---

## 1. Core Methodology & Optimizations

Delivering a smooth, real-time computer vision demo requires balancing heavy deep learning inference with UI rendering. The script employs several key strategies to achieve high FPS and stable visual feedback.

### Fast Preprocessing & Inference Architecture
Instead of relying on standard `PIL` images and `torchvision.transforms` (which are notoriously slow for real-time video), the pipeline uses pure **NumPy and OpenCV**:
*   The fretboard crop is directly resized to 229×229 (the expected input size for the Inception-ResNet V2 backbone) using `cv2.resize`.
*   Normalization to `[-1, 1]` is performed using matrix operations in NumPy/Torch before passing to the device (MPS/CUDA/CPU).
*   Inference is wrapped in `torch.inference_mode()` (faster than `no_grad`), and if using PyTorch 2.0+, the model is automatically optimized with `torch.compile()` to yield a 20-40% speed boost.

### Optimized Roboflow Fretboard Detection
Querying a local Roboflow Docker server (`localhost:9001`) for bounding boxes is the heaviest operation. To eliminate lag:
*   **Zero Disk I/O**: The frame is passed as an in-memory NumPy array directly to the Roboflow SDK, bypassing the slow process of writing and reading temporary JPEG files.
*   **Downscaling**: Frames are dynamically downscaled to a maximum width of 640px (`DETECT_MAX_W`) before detection. The resulting bounding box coordinates are then scaled back to the original resolution.
*   **Throttling**: The fretboard location rarely jumps instantly, so detection only runs once every 6 frames (`ROBOFLOW_EVERY = 6`). However, the smaller cropped region is *classified* on every single frame (`INFER_EVERY_N = 1`) for instantaneous chord updates.

### Temporal Smoothing & Stability
Raw neural network predictions frame-by-frame can be erratic, especially while a player's hand is shifting between chords.
*   **Bounding Box IIR Filter**: A low-pass filter (`smooth_box = beta * new + (1-beta) * old`) prevents the drawn fretboard rectangle from jittering.
*   **EMA Probability Smoothing**: Exponential Moving Average (`alpha = 0.35`) is applied to the class probability distribution. This completely eliminates flickering while maintaining responsiveness.
*   **Information Entropy (Uncertainty Estimation)**: Rather than just looking at the highest probability class, the script calculates the **Shannon entropy** of the tracked probability distribution. If the entropy exceeds a set threshold (e.g., the model is evenly split between two chords during a transition), the system forces an "Uncertain" (`?`) state, hiding wildly incorrect guesses.

### Visual Polish
The right-hand side of the screen renders a dedicated Dashboard panel utilizing purely OpenCV drawing primitives (no external GUI libraries). It features:
*   A procedural vertical gradient background.
*   Glowing text (rendered by drawing the string multiple times with increasing thickness and decreasing opacity).
*   Lerped (Linearly Interpolated) probability bars that smoothly glide to their target values.
*   A circular confidence arc and a scrolling timeline of the last 14 chords played.

---

## 2. Function Reference

Below is a breakdown of the primary functions and classes used in the execution pipeline.

### Model & Neural Network
*   **`GuitarChordClassifier(nn.Module)`**
    The PyTorch model definition. It utilizes the `timm` library to load an `inception_resnet_v2` backbone. The standard classifier head is replaced with a custom MLP (AdaptiveAvgPool2d → Linear(256) → Dropout → Linear(128) → Dropout → Output(5)).
*   **`load_model(weights_path, device)`**
    Instantiates the classifier, loads the `.pth` weights, switches to `eval()` mode, and attempts to apply `torch.compile()`.

### Image Processing & Inference
*   **`preprocess_crop(crop_bgr, device)`**
    Takes the raw BGR OpenCV crop, converts to RGB, resizes, normalizes `[0, 255] -> [-1.0, 1.0]`, and returns a PyTorch Tensor ready for the model.
*   **`classify_crop(crop_bgr, model, device)`**
    Executes the forward pass. Applies `Softmax` and returns the name of the winning class, its confidence, and a dictionary of all class probabilities.

### Bounding Box Detection
*   **`build_roboflow_client()`**
    Initializes the `InferenceHTTPClient` connecting to the defined server address.
*   **`detect_fretboard(client, frame_bgr)`**
    Handles the downscaling of the frame and the API call to Roboflow. Implements a fallback to a temp-file approach if the direct NumPy array fails (for older SDK compatibility). Returns physical coordinates `(x1, y1, x2, y2)`.
*   **`crop_with_padding(frame, box, pad)`**
    Expands the detected bounding box by a relative padding scalar (default 15%) to ensure the entire hand and strings fit comfortably inside the classifier's view.

### Smoothing Utilities
*   **`smooth_box(new_box, prev_box, beta)`**
    Infinte Impulse Response (IIR) tracking for bounding boxes.
*   **`smooth_probs(new_probs, prev_smoothed, alpha)`**
    Exponential Moving Average tracking for the output probability distribution. Returns a blended probability dictionary.
*   **`is_uncertain(probs_dict)`** *(Inline inside `run()`)*
    Calculates the normalized entropy of the probability distribution. Returns `True` if the entropy is high (e.g., probabilities are spread out evenly rather than collapsing onto a single confident guess).

### Drawing & UI Components
*   **`_gradient_panel(h, w)`**
    Generates a dark background image fading vertically.
*   **`_glow_text(img, text, pos, font, scale, colour, thickness, layers)`**
    Creates a simulated neon-glow effect behind OpenCV text strings.
*   **`_confidence_ring(img, cx, cy, radius, confidence, colour, thick)`**
    Draws a semi-circular progress bar using `cv2.ellipse`.
*   **`draw_fretboard_box(frame, box, colour, confidence, fidx)`**
    Draws the live tracking rectangle on the video feed. Includes a sine-wave pulse effect based on the frame index (`fidx`), a semi-transparent color tint using `cv2.addWeighted`, and custom thick corner borders.
*   **`draw_title_bar(frame, status, fps, fidx)`**
    Renders top-screen metrics including smoothed FPS and a flashing red recording indicator.
*   **`draw_chord_panel(...)`**
    The main UI assembly routine. Calls the gradient panel, glow text, and confidence ring. Additionally manages the linear interpolation (`BAR_LERP_SPEED`) of the horizontal class probability bars, preventing jarring snappy updates. Also handles drawing the mini colored blocks forming the "History" timeline strip.

### Orchestration
*   **`run(source, use_roboflow, save_output, output_path)`**
    The primary video capture loop.
    1. Grabs a frame from `cv2.VideoCapture`.
    2. Runs `detect_fretboard` if it's the Nth frame.
    3. Crops the image.
    4. Runs `classify_crop`.
    5. Calculates EMA and Information Entropy to decide the final chord label string.
    6. Updates logic for visual transition flashes.
    7. Dispatches all `draw_*` functions onto the canvas.
    8. Shows the frame via `cv2.imshow` and optionally writes to disk using `cv2.VideoWriter`.
