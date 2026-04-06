from inference_sdk import InferenceHTTPClient
import supervision as sv
import cv2


IMAGE_PATH = "tt.jpeg"


image = cv2.imread(IMAGE_PATH)

ROBOFLOW_API_KEY = "[ENCRYPTION_KEY]"

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY,
)

result = client.run_workflow(
    workspace_name="test-kpcsq",
    workflow_id="find-fretboards",
    images={
        "image": IMAGE_PATH, 
    },
    use_cache=True,
)

detections = sv.Detections.from_inference(result[0]["predictions"])

EXTRA_RELATIVE_PADDING = 0.1

for i, box in enumerate(detections.xyxy):
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]

    pad_x = int((x2 - x1) * EXTRA_RELATIVE_PADDING)
    pad_y = int((y2 - y1) * EXTRA_RELATIVE_PADDING)

    crop = image[
        max(0, y1 - pad_y) : min(h, y2 + pad_y),
        max(0, x1 - pad_x) : min(w, x2 + pad_x)
    ]
    sv.plot_image(crop)
    cv2.imwrite(f"crop_{i}.jpg", crop)
    print(f"Saved crop_{i}.jpg")