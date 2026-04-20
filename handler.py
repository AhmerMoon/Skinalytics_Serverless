import runpod
from ultralytics import YOLO
import torch
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import cv2
import numpy as np
import base64
import io

# ==========================================
# 1. LOAD ALL MODELS GLOBALLY
# ==========================================
print("🧠 Loading All Models...")

try:
    m0_model = resnet50()
    num_ftrs = m0_model.fc.in_features
    m0_model.fc = torch.nn.Linear(num_ftrs, 3) 
    m0_model.load_state_dict(torch.load("models/M0.pth", map_location=torch.device('cpu')))
    m0_model.eval()
    print("[+] M0 Loaded Successfully!")
except Exception as e:
    print(f"[!] M0 Load Error: {e}")

print("Loading M1, M2, M3...")
m1_model = YOLO("models/M1.pt")
m2_model = YOLO("models/M2.pt")
m3_model = YOLO("models/M3.pt")

# ==========================================
# 2. CLASS MAPPINGS & COLORS
# ==========================================
M0_CLASSES = ["Dry", "Normal", "Oily"] 
M1_NAMES = {0: "Acne", 1: "Blackhead"}
M2_NAMES = {0: "Wrinkles", 1: "Rosacea", 2: "Hyperpigmentation", 3: "Dark Circle", 4: "M2_Scar"}
M3_NAMES = {0: "Scar", 1: "Freckles", 2: "Eczema", 3: "Pores"}

M1_COLORS = {0: (0, 0, 255), 1: (255, 0, 0)} 
M2_COLORS = {0: (0, 255, 255), 1: (0, 165, 255), 2: (128, 0, 128), 3: (255, 255, 0), 4: (0, 255, 0)} 
M3_COLORS = {0: (50, 205, 50), 1: (203, 192, 255), 2: (0, 100, 255), 3: (200, 200, 200)}

COLOR_NAMES = {
    "Acne": "Red", "Blackhead": "Blue", "Wrinkles": "Yellow", "Rosacea": "Orange", 
    "Hyperpigmentation": "Purple", "Dark Circle": "Cyan", "M2_Scar": "Green", 
    "Scar": "Light Green", "Freckles": "Pinkish Purple", "Eczema": "Dark Orange", "Pores": "Light Gray"
}

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def calculate_scores(condition_name, count, total_area):
    severity = 0
    if condition_name in ["Acne", "Blackhead", "Scar", "M2_Scar"]:
        if count <= 3 and total_area < 5000: severity = 20
        elif count <= 8 and total_area < 20000: severity = 50
        elif count <= 15: severity = 75
        else: severity = 95
    elif condition_name in ["Freckles", "Pores"]:
        if count <= 15: severity = 30
        elif count <= 40: severity = 60
        else: severity = 85
    else: 
        if total_area < 15000: severity = 35
        elif total_area < 50000: severity = 65
        elif total_area < 100000: severity = 85
        else: severity = 98
        
    severity = min(100, severity + min(5, int(count * 0.5)))
    health = max(0, 100 - severity)
    return health, severity

def get_facial_crops(image):
    h, w = image.shape[:2]
    return [
        (image[0:int(h*0.35), 0:w], 0, 0), 
        (image[int(h*0.35):int(h*0.75), 0:int(w*0.45)], 0, int(h*0.35)),
        (image[int(h*0.35):int(h*0.75), int(w*0.55):w], int(w*0.55), int(h*0.35)),
        (image[int(h*0.40):int(h*0.70), int(w*0.30):int(w*0.70)], int(w*0.30), int(h*0.40)),
        (image[int(h*0.70):h, int(w*0.20):int(w*0.80)], int(w*0.20), int(h*0.70)),
        (image, 0, 0)
    ]

def predict_and_map_crops(image_bgr, model, conf_thresh, model_id):
    crops = get_facial_crops(image_bgr)
    all_boxes = []
    for crop_img, x_offset, y_offset in crops:
        results = model.predict(source=crop_img, conf=conf_thresh, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            all_boxes.append([x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset, 
                              float(box.conf[0]), int(box.cls[0]), model_id])
    return all_boxes

def calculate_iou(box1, box2):
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

# ==========================================
# 4. RUNPOD HANDLER FUNCTION
# ==========================================
def handler(event):
    input_data = event.get("input", {})
    image_b64 = input_data.get("image", None)
    
    if not image_b64:
        return {"status": 400, "error": "No base64 image provided"}

    # Base64 Decode
    img_bytes = base64.b64decode(image_b64)
    img_nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- A. M0 Classification ---
    skin_type = "Unknown"
    try:
        pil_img = Image.fromarray(img_rgb)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            outputs = m0_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            skin_type = M0_CLASSES[predicted.item()]
    except Exception as e:
        skin_type = "Error"

    # --- B. Object Detection ---
    boxes_m1 = predict_and_map_crops(img_bgr, m1_model, conf_thresh=0.25, model_id=1)
    boxes_m2 = predict_and_map_crops(img_bgr, m2_model, conf_thresh=0.25, model_id=2)
    boxes_m3 = predict_and_map_crops(img_bgr, m3_model, conf_thresh=0.25, model_id=3)
    
    def apply_nms(boxes, iou_thresh=0.4):
        if not boxes: return []
        b_tensor = torch.tensor([b[:4] for b in boxes], dtype=torch.float32)
        s_tensor = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
        keep = torchvision.ops.nms(b_tensor, s_tensor, iou_threshold=iou_thresh)
        return [boxes[i] for i in keep]

    all_raw_boxes = apply_nms(boxes_m1) + apply_nms(boxes_m2) + apply_nms(boxes_m3)

    # --- C. Cross-Model Filtering ---
    final_boxes = []
    freckle_boxes = [b for b in all_raw_boxes if b[6] == 3 and b[5] == 1] 
    for box in all_raw_boxes:
        if box[6] == 2 and box[5] == 2: # M2 Hyperpigmentation
            is_overlap = any(calculate_iou(box[:4], fb[:4]) > 0.4 for fb in freckle_boxes)
            if not is_overlap: final_boxes.append(box)
        else:
            final_boxes.append(box)

    # --- D. Aggregating Data & Drawing Image ---
    tech_data_temp = {}
    output_img = img_bgr.copy() # Canvas tayar karo
    
    for box in final_boxes:
        x1, y1, x2, y2, conf, cls_id, model_id = box
        area = int((x2 - x1) * (y2 - y1))
        
        if model_id == 1: 
            name, color = M1_NAMES.get(cls_id, 'Unknown'), M1_COLORS.get(cls_id, (255,255,255))
        elif model_id == 2: 
            name, color = M2_NAMES.get(cls_id, 'Unknown'), M2_COLORS.get(cls_id, (255,255,255))
        else: 
            name, color = M3_NAMES.get(cls_id, 'Unknown'), M3_COLORS.get(cls_id, (255,255,255))
            
        if name not in tech_data_temp:
            tech_data_temp[name] = {"count": 0, "total_area": 0, "color_name": COLOR_NAMES.get(name, "Unknown"), "instances": []}
            
        tech_data_temp[name]["count"] += 1
        tech_data_temp[name]["total_area"] += area
        tech_data_temp[name]["instances"].append({
            "conf": round(conf, 3), "area_px": area, 
            "coords": [int(x1), int(y1), int(x2), int(y2)]
        })

        # Bounding box aur label draw karna
        label = f"{name} {conf:.2f}"
        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(output_img, (int(x1), int(y1) - 15), (int(x1) + w, int(y1)), color, -1)
        cv2.putText(output_img, label, (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # M0 HUD Draw karna
    hud_text = f" Skin Type: {skin_type} "
    (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.rectangle(output_img, (15, 15), (15 + tw + 20, 15 + th + 20), (0, 0, 0), -1) 
    cv2.putText(output_img, hud_text, (25, 15 + th + 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # --- E. Convert Processed Image back to Base64 ---
    _, buffer = cv2.imencode('.jpg', output_img)
    processed_b64 = base64.b64encode(buffer).decode('utf-8')

    # --- F. Build Hybrid Enterprise JSON ---
    report_data = {
        "status": 200,
        "task_status": "success",
        "data": {
            "metadata": {
                "image_resolution": f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
                "skin_type": skin_type
            },
            "processed_image_base64": processed_b64, # <-- IMAGE YAHAN JAYEGI!
            "results": []
        }
    }
    
    for cond_name, data in tech_data_temp.items():
        health_sc, severity_sc = calculate_scores(cond_name, data["count"], data["total_area"])
        report_data["data"]["results"].append({
            "type": cond_name,
            "health_score": health_sc,     
            "severity_score": severity_sc, 
            "ui_color": data["color_name"],
            "technical_details": {
                "total_instances": data["count"],
                "total_affected_area_px": data["total_area"],
                "boxes": data["instances"]
            }
        })

    return report_data

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})