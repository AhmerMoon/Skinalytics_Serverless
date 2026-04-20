import base64
import json
from handler import handler

# 1. Apni test image ka naam
IMAGE_PATH = "test_face.jpg" 

print("[*] Encoding image to Base64...")
try:
    with open(IMAGE_PATH, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
except FileNotFoundError:
    print(f"[!] Error: {IMAGE_PATH} nahi mili.")
    exit()

mock_event = {
    "input": {
        "image": b64_string
    }
}

# 2. Call the AI Engine
print("\n[*] Processing Image through M0, M1, M2, M3... Please wait...")
response = handler(mock_event)

# ==========================================
# 3. MAGIC: Save the Image to Folder
# ==========================================
try:
    output_b64 = response["data"]["processed_image_base64"]
    # Base64 string ko wapis bytes me convert karo
    img_bytes = base64.b64decode(output_b64)
    
    # Folder me save karo
    with open("output_preview.jpg", "wb") as f:
        f.write(img_bytes)
    print("\n[+] 📸 BOOM! 'output_preview.jpg' saved in your folder! Usay open kar k check karo.")
    
    # Terminal ko flood hone se bachanay k liye JSON se lamba text hata do
    response["data"]["processed_image_base64"] = "[... MASSIVE_BASE64_STRING_HIDDEN_FOR_CONSOLE ...]"
    
except KeyError:
    print("\n[!] Error: Base64 image not found in response.")

# 4. Print Clean JSON
print("\n[+] Enterprise JSON Report:\n")
print(json.dumps(response, indent=4))