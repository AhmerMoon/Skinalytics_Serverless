import modal
import time
from fastapi import FastAPI, Request

app = modal.App("skinalytics-api")

env_image = (
    modal.Image.from_registry("ahmermoon/skinalytics-backend:latest")
    .pip_install("fastapi", "numpy<2.0.0", "ultralytics", "opencv-python-headless", "torchvision") 
)

@app.cls(gpu="T4", image=env_image, timeout=200)
class SkinalyticsEngine:
    
    @modal.enter()
    def load_models(self):
        print("🧠 [GPU Engine] Booting GPU & Loading Models...")
        import torch
        from torchvision.models import resnet50
        from ultralytics import YOLO
        import os
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_path = "models"
        
        # Load M0 (ResNet50)
        try:
            self.m0_model = resnet50()
            self.m0_model.fc = torch.nn.Linear(self.m0_model.fc.in_features, 3) 
            self.m0_model.load_state_dict(torch.load(f"{base_path}/M0.pth", map_location='cpu'))
            self.m0_model.to(self.device)
            self.m0_model.eval()
            self.M0_CLASSES = ["Dry", "Normal", "Oily"] 
        except Exception as e:
            print(f"❌ Error loading M0: {e}")
            self.m0_model = None

        # Load YOLO Models
        try:
            self.m1_model = YOLO(f"{base_path}/M1.pt")
            self.m2_model = YOLO(f"{base_path}/M2.pt")
            self.m3_model = YOLO(f"{base_path}/M3.pt")
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            
        print("✅ [GPU Engine] All Systems Ready!")

    @modal.method()
    def prewarm(self):
        """Triggered by Orchestrator after user login to mitigate cold starts"""
        print(f"🔥 Pre-warm triggered at {time.time()}. GPU Active.")
        return {"status": 200, "message": "GPU is warm"}

    @modal.method()
    def analyze(self, payload: dict):
        import base64
        import cv2
        import numpy as np
        import torch
        import torchvision
        from torchvision import transforms
        from PIL import Image

        image_b64 = payload.get("image")
        if not image_b64:
            return {"status": 400, "error": "No image provided"}

        img_bytes = base64.b64decode(image_b64)
        img_nparr = np.frombuffer(img_bytes, np.uint8)
        raw_bgr = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)
        img_bgr = raw_bgr.copy()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ==========================================
        # STEP 1: SKIN TYPE (M0)
        # ==========================================
        skin_type = "Unknown"
        if self.m0_model:
            try:
                pil_img = Image.fromarray(img_rgb)
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                input_tensor = transform(pil_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.m0_model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    skin_type = self.M0_CLASSES[predicted.item()]
            except: pass

        # ==========================================
        # STEP 2: YOLO INFERENCE & PATCHING
        # ==========================================
        def get_crops(image):
            h, w = image.shape[:2]
            return [
                (image[0:int(h*0.35), 0:w], 0, 0),
                (image[int(h*0.35):int(h*0.75), 0:int(w*0.45)], 0, int(h*0.35)),
                (image[int(h*0.35):int(h*0.75), int(w*0.55):w], int(w*0.55), int(h*0.35)),
                (image[int(h*0.40):int(h*0.70), int(w*0.30):int(w*0.70)], int(w*0.30), int(h*0.40)),
                (image[int(h*0.70):h, int(w*0.20):int(w*0.80)], int(w*0.20), int(h*0.70)),
                (image, 0, 0)
            ]

        def get_boxes(model, model_id):
            if not model: return []
            all_b = []
            img_area = img_bgr.shape[0] * img_bgr.shape[1]
            for crop_img, x_off, y_off in get_crops(img_bgr):
                # ✅ CONFIDENCE REDUCED TO 0.20 FOR MORE DETECTIONS
                res = model.predict(source=crop_img, conf=0.20, verbose=False) 
                for box in res[0].boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                                            
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # ✅ DARK CIRCLE GIANT BOX FILTER (Discards boxes larger than 8% of the image)
                    if model_id == 2 and cls_id == 3:
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area > (img_area * 0.08):
                            continue
                            
                    all_b.append([x1+x_off, y1+y_off, x2+x_off, y2+y_off, conf, cls_id, model_id])
            return all_b

        all_boxes = get_boxes(self.m1_model, 1) + get_boxes(self.m2_model, 2) + get_boxes(self.m3_model, 3)
        
        # Batched NMS to remove duplicates across overlapping patches
        final_boxes = []
        if all_boxes:
            b_tensor = torch.tensor([b[:4] for b in all_boxes], dtype=torch.float32).to(self.device)
            s_tensor = torch.tensor([b[4] for b in all_boxes], dtype=torch.float32).to(self.device)
            c_tensor = torch.tensor([b[6] * 10 + b[5] for b in all_boxes], dtype=torch.float32).to(self.device)
            
            keep = torchvision.ops.batched_nms(b_tensor, s_tensor, c_tensor, iou_threshold=0.4)
            final_boxes = [all_boxes[i] for i in keep.cpu().numpy()]

        # ==========================================
        # STEP 3: CONSOLIDATION & DIMINISHING SCORING
        # ==========================================
        ALL_CONDS = ["Acne", "Blackhead", "Wrinkles", "Rosacea", "Hyperpigmentation", 
                     "Dark Circle", "Scar", "Freckles", "Eczema", "Pores"]
        
        tech_data = {c: {"count": 0, "area": 0} for c in ALL_CONDS}
        
        M_NAMES = {
            1: {0: "Acne", 1: "Blackhead"}, 
            2: {0: "Wrinkles", 1: "Rosacea", 2: "Hyperpigmentation", 3: "Dark Circle", 4: "Scar"}, 
            3: {0: "Scar", 1: "Freckles", 2: "Eczema", 3: "Pores"}
        }

        LABEL_COLORS = {
            "Acne": (0, 0, 255),               
            "Blackhead": (80, 80, 80),         
            "Wrinkles": (255, 0, 255),         
            "Rosacea": (147, 20, 255),         
            "Hyperpigmentation": (0, 100, 255),
            "Dark Circle": (255, 0, 0),        
            "Scar": (250, 206, 135),           
            "Freckles": (0, 165, 255),         
            "Eczema": (50, 205, 50),           
            "Pores": (255, 255, 0)             
        }

        output_img = img_bgr.copy()
        
        for box in final_boxes:
            x1, y1, x2, y2, conf, cls_id, m_id = box
            name = M_NAMES[m_id].get(cls_id)
            if name and name in tech_data:
                tech_data[name]["count"] += 1
                tech_data[name]["area"] += int((x2-x1)*(y2-y1))
                
                color = LABEL_COLORS.get(name, (0, 255, 255))
                cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # ✅ ADDED CONFIDENCE VALUE TO THE LABEL TEXT
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                cv2.rectangle(output_img, (int(x1), int(y1) - th - 6), (int(x1) + tw, int(y1)), color, -1)
                
                text_color = (0, 0, 0) if name in ["Pores", "Scar", "Freckles"] else (255, 255, 255)
                cv2.putText(output_img, label, (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        # TOP LEFT SKIN TYPE TAG
        tag_text = f"Skin Type: {skin_type}"
        (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(output_img, (10, 10), (10 + tw + 20, 10 + th + 20), (0, 0, 0), -1)
        cv2.putText(output_img, tag_text, (20, 10 + th + 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # The "Diminishing Returns" Fix for Scoring
        def calc_raw_score(name, count, area):
            if count == 0: return 10.0
            
            severe = ["Acne", "Rosacea", "Eczema", "Scar"]
            mild = ["Freckles", "Pores", "Blackhead"]
            
            penalty = 0.0
            p1 = 0.7 if name in severe else (0.2 if name in mild else 0.4)
            p2 = p1 * 0.5  
            p3 = p1 * 0.2  
            
            for i in range(1, count + 1):
                if i <= 3: penalty += p1
                elif i <= 8: penalty += p2
                else: penalty += p3
                
            penalty = min(penalty, 7.0)
            penalty += min(area / 25000.0, 1.5)
            
            final_raw = 10.0 - penalty
            return max(1.5, round(final_raw, 1))

        results_list = []
        raw_conditions_map = {}
        total_score_10 = 0.0

        for cond in ALL_CONDS:
            cnt = tech_data[cond]["count"]
            area = tech_data[cond]["area"]
            
            raw_score = calc_raw_score(cond, cnt, area)
            total_score_10 += raw_score
            ui_score = int(raw_score * 10) 

            raw_conditions_map[cond] = {"count": cnt, "area": area, "score": raw_score}
            
            results_list.append({
                "name": cond,                          
                "score": raw_score,                    
                "type": cond.lower().replace(" ", "_"),
                "ui_score": ui_score,                  
                "raw_score": raw_score,                
                "detections_count": cnt,
                "total_area_pixels": area,
                "severity": "Good" if ui_score >= 80 else "Warning" if ui_score >= 60 else "Critical"
            })

        overall_health_score_100 = int((total_score_10 / 10.0) * 10)

        _, buffer = cv2.imencode('.jpg', output_img)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": 200, 
            "data": {
                "skin_type": skin_type, 
                "overall_score_100": overall_health_score_100,
                "processed_image_base64": processed_b64, 
                "results_list": results_list,
                "raw_conditions_map": raw_conditions_map
            }
        }

web_app = FastAPI()

@web_app.post("/analyze") 
async def process_endpoint(request: Request):
    payload = await request.json()
    engine = SkinalyticsEngine()
    return await engine.analyze.remote.aio(payload)

@web_app.get("/prewarm")
async def trigger_prewarm():
    print("🔥 [GPU Engine] Boot sequence started via /prewarm")
    engine = SkinalyticsEngine()
    await engine.prewarm.remote.aio()
    return {"status": 200, "message": "GPU Warmed up"}

@app.function(image=env_image)
@modal.asgi_app()
def fastapi_app():
    return web_app