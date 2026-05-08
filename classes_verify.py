import modal

app = modal.App("model-verifier")

env_image = (
    modal.Image.from_registry("ahmermoon/skinalytics-backend:latest")
    .pip_install("ultralytics")
)

@app.function(gpu="T4", image=env_image)
def verify_classes():
    from ultralytics import YOLO
    import os
    
    base_path = "models"
    
    print("\n" + "="*50)
    print("🕵️ MODEL VERIFICATION START")
    print("="*50 + "\n")
    
    for m_name in ["M1.pt", "M2.pt", "M3.pt"]:
        model_path = f"{base_path}/{m_name}"
        
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"[+] {m_name} Loaded Successfully!")
            print(f"[*] Classes baked inside {m_name}: {model.names}\n")
        else:
            print(f"[!] ERROR: {m_name} file missing inside Docker container!\n")
            
    print("="*50)
    print("🏁 VERIFICATION COMPLETE")
    print("="*50 + "\n")

# ⚡ THE FIX: Naya Modal execution tareeqa
@app.local_entrypoint()
def main():
    verify_classes.remote()