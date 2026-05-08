import requests
import base64
import json
import time

URL = "https://skinalyticsapp--skinalytics-api-fastapi-app.modal.run/analyze"
IMAGE_PATH = "test_face.jpg"

print("[*] Image ko Base64 me convert kar raha hu...")
try:
    with open(IMAGE_PATH, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
except FileNotFoundError:
    print(f"[!] Error: {IMAGE_PATH} nahi mili.")
    exit()

payload = {"image": b64_string}
print("[*] Modal Cloud par request jaa rahi hai... (Fingers crossed 🤞)")
start_time = time.time()

try:
    response = requests.post(URL, json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        print(f"\n[+] SUCCESS! Cloud ne {round(end_time - start_time, 2)} seconds me response diya!")
        ai_output = response.json()
        
        # ----- Image save karo (pehle processed_image_base64 check karo, nahi to annotated_image_base64)
        img_base64 = None
        if "data" in ai_output:
            if "processed_image_base64" in ai_output["data"]:
                img_base64 = ai_output["data"]["processed_image_base64"]
            elif "annotated_image_base64" in ai_output["data"]:
                img_base64 = ai_output["data"]["annotated_image_base64"]
        
        if img_base64:
            img_bytes = base64.b64decode(img_base64)
            with open("modal_final_output.jpg", "wb") as f:
                f.write(img_bytes)
            print("[+] 📸 Labeled Image 'modal_final_output.jpg' k naam se save ho gayi hai.")
        else:
            print("[!] Warning: Image data not found in response.")
        
        # ----- Print JSON with base64 hidden (dono fields ko hide karo)
        if "data" in ai_output:
            if "processed_image_base64" in ai_output["data"]:
                ai_output["data"]["processed_image_base64"] = "[... HIDDEN ...]"
            if "annotated_image_base64" in ai_output["data"]:
                ai_output["data"]["annotated_image_base64"] = "[... HIDDEN ...]"
        
        print("\n[+] Final Cloud JSON:\n", json.dumps(ai_output, indent=4))
    else:
        print(f"\n[!] Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"[!] Request Failed: {e}")