import os
import modal
import json
import re
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

modal_app = modal.App("skinalytics-main-backend")
env_image = modal.Image.debian_slim().pip_install(
    "fastapi", "httpx", "pydantic", "openai"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ IMPORTANT: Yahan apne GPU Model wali file ka URL lazmi replace karna
GPU_ENGINE_URL = "https://skinalyticsapp--skinalytics-api-fastapi-app.modal.run"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
KLAVIYO_PRIVATE_KEY = os.environ.get("KLAVIYO_PRIVATE_KEY")
LIST_ID = "XRyUfY"

# ==========================================
# PYDANTIC MODELS
# ==========================================
class SkinRequest(BaseModel):
    image_base64: str

class FinalReportRequest(BaseModel):
    skin_analysis: Dict[str, Any]
    lifestyle_data: Dict[str, Any]
    user_preferences: Dict[str, Any]

class SubscribeRequest(BaseModel):
    email: str

class TrackRequest(BaseModel):
    email: str
    event: str
    properties: Optional[Dict] = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'__', '', text)
    text = re.sub(r'_', '', text)
    return text

def generate_product_search_query(skin_type: str, raw_conditions: dict) -> str:
    """Generate a search query for Shopify store"""
    keywords = []
    if skin_type == "Dry":
        keywords.extend(["hydrating", "moisturizing"])
    elif skin_type == "Oily":
        keywords.extend(["oil control", "matte"])
    elif skin_type == "Normal":
        keywords.extend(["gentle", "balance"])

    for condition, info in raw_conditions.items():
        cnt = info.get("count", 0)
        if cnt == 0:
            continue
        cond_lower = condition.lower()
        if "acne" in cond_lower or "blackhead" in cond_lower:
            keywords.extend(["acne", "clarifying"])
        if "wrinkle" in cond_lower:
            keywords.extend(["anti aging", "wrinkle"])
        if "rosacea" in cond_lower or "redness" in cond_lower:
            keywords.extend(["calming", "redness"])
        if "pigmentation" in cond_lower or "dark circle" in cond_lower:
            keywords.extend(["brightening", "hyperpigmentation"])
        if "scar" in cond_lower:
            keywords.append("repair")
        if "freckles" in cond_lower or "eczema" in cond_lower or "pores" in cond_lower:
            keywords.extend(["gentle", "soothing"])

    unique = list(dict.fromkeys(keywords))
    return " ".join(unique[:3]) if unique else "skincare"

# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/")
def health():
    return {"status": "OK", "message": "Skinalytics Main Orchestrator is running"}

@app.get("/warmup")
async def warmup():
    """
    Triggered by Flutter after user login. 
    Wakes up the GPU engine in the background to prevent cold-start delays.
    """
    print("🔥 [Orchestrator] Triggering GPU pre-warm in background...")
    async def call_gpu():
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                await client.get(f"{GPU_ENGINE_URL}/prewarm")
        except Exception as e:
            print(f"⚠️ Pre-warm call warning: {e}")
            
    asyncio.create_task(call_gpu())
    return {"status": 202, "message": "GPU Boot sequence initiated"}

@app.post("/analyze")
async def analyze(req: SkinRequest):
    print("[*] Starting analysis workflow...")
    try:
        timeout = httpx.Timeout(150.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            model_res = await client.post(f"{GPU_ENGINE_URL}/analyze", json={"image": req.image_base64})

        if model_res.status_code != 200:
            raise HTTPException(status_code=500, detail=f"GPU Model returned {model_res.status_code}")

        resp_json = model_res.json()
        if resp_json.get("status") != 200:
            raise HTTPException(status_code=500, detail=f"Modal error: {resp_json.get('error')}")

        data = resp_json.get("data", {})
        
        # Formatting exactly as Flutter (report_screen.dart) expects
        response_data = {
            "success": True,
            "skin_type": data.get("skin_type", "Unknown"),
            "processed_image": data.get("processed_image_base64"),
            "results": data.get("results_list", []),          # Hybrid YouCam list
            "raw_conditions": data.get("raw_conditions_map", {}), # Counts and areas
            "overall_score": data.get("overall_score_100", 65.0), # 0-100 format
            "generated_at": datetime.now().isoformat()
        }

        print(f"[+] Analysis done. Skin: {response_data['skin_type']}, Overall: {response_data['overall_score']}/100")
        return response_data

    except Exception as e:
        print(f"[!] Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/final-report")
async def final_report(req: FinalReportRequest):
    skin_type = "Unknown"
    overall_score = 65.0  
    cond_text = "- No significant conditions detected."
    
    try:
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        openai_client = OpenAI(api_key=api_key)

        # Extract data safely from Flutter's payload
        skin_analysis = req.skin_analysis or {}
        skin_type = skin_analysis.get("skin_type", "Unknown")
        
        # Pulling the exact overall score calculated by Flutter / Backend
        overall_score = skin_analysis.get("overall_health", 65.0) 
        scores_list = skin_analysis.get("condition_scores", [])
        
        raw_data = skin_analysis.get("raw_data", {})
        raw_conditions = raw_data.get("raw_conditions", {})
        if skin_type == "Unknown":
            skin_type = raw_data.get("skin_type", "Unknown")

        cond_text = ""
        for name, info in raw_conditions.items():
            cnt = info.get("count", 0)
            if cnt > 0:
                cond_text += f"- {name}: detected {cnt} time(s)\n"
                
        if not cond_text:
            cond_text = "- No significant conditions detected. Skin appears mostly clear."

        lifestyle = req.lifestyle_data
        prefs = req.user_preferences
        search_query = generate_product_search_query(skin_type, raw_conditions)

        print(f"[*] Requesting GPT Report. Skin: {skin_type}, Score: {overall_score}/100")

        prompt = f"""
You are an expert dermatologist. Generate a {prefs.get('report_type', 'full')} skin analysis report in {prefs.get('language', 'English')}.

REQUIRED SECTIONS (plain text, no markdown like ** or __):
- Overview
- Skin Health Score Explanation (The overall score is {overall_score}/100. 100 is flawless, 0 is severe)
- Detailed Concern Analysis (Based exactly on the conditions listed below)
- Lifestyle Impact
- Recommendations
- Product Suggestions (if {prefs.get('include_products', True)})
- Skincare Routine (if {prefs.get('include_routine', True)})

Skin Type: {skin_type}
Detected Conditions (Counts):
{cond_text}

Individual Condition Scores (Out of 10. 10=flawless, 0=severe):
{json.dumps(scores_list, indent=2)}

Lifestyle Data:
{json.dumps(lifestyle, indent=2)}

Write in clear, plain text. Be thorough but concise. Do not use asterisks or underscores for formatting.
"""
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            max_tokens=1200,
            messages=[
                {"role": "system", "content": "You are a professional dermatologist."},
                {"role": "user", "content": prompt},
            ],
        )
        report_text = clean_markdown(response.choices[0].message.content)
        return {"success": True, "report": report_text, "product_search_query": search_query}

    except Exception as e:
        print(f"Final report error: {e}")
        fallback = f"""
SKIN ANALYSIS REPORT

Overview:
Your skin analysis has been completed based on the provided image and lifestyle details.

Skin Type: {skin_type}
Overall Score: {overall_score}/100

Detected Conditions:
{cond_text}

Lifestyle Impact:
- Sleep: {req.lifestyle_data.get('sleep_hours', 'N/A')} hours
- Water: {req.lifestyle_data.get('water_glasses', 'N/A')} glasses
- Sun Exposure: {req.lifestyle_data.get('sun_exposure', 'N/A')}
- Stress Level: {req.lifestyle_data.get('stress_level', 'N/A')}
- Skin Sensitivity: {req.lifestyle_data.get('skin_sensitivity', 'N/A')}

Recommendations:
1. Cleanse twice daily.
2. Apply SPF 30+ every morning.
3. Use a suitable moisturizer.
4. Stay hydrated.

Skincare Routine:
Morning: Cleanse -> Moisturize -> SPF
Evening: Double cleanse -> Treatment -> Moisturize

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Powered by Skinalytics AI
"""
        search_query = generate_product_search_query(skin_type, {})
        return {"success": True, "report": fallback, "product_search_query": search_query}

# ==========================================
# KLAVIYO INTEGRATIONS
# ==========================================
@app.post("/subscribe")
async def subscribe(request: SubscribeRequest):
    headers = {
        "Authorization": f"Klaviyo-API-Key {KLAVIYO_PRIVATE_KEY}",
        "revision": "2024-10-15",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient() as client:
        check_url = f"https://a.klaviyo.com/api/lists/{LIST_ID}/profiles/"
        params = {"filter": f'equals(email,"{request.email}")'}
        check_resp = await client.get(check_url, headers=headers, params=params)
        if check_resp.status_code == 200:
            data = check_resp.json().get("data", [])
            if data:
                return {"status": "already_subscribed"}

        url = "https://a.klaviyo.com/api/profile-subscription-bulk-create-jobs/"
        payload = {
            "data": {
                "type": "profile-subscription-bulk-create-job",
                "attributes": {
                    "profiles": {
                        "data": [{
                            "type": "profile",
                            "attributes": {
                                "email": request.email,
                                "subscriptions": {
                                    "email": {"marketing": {"consent": "SUBSCRIBED"}}
                                },
                            },
                        }]
                    }
                },
                "relationships": {
                    "list": {"data": {"type": "list", "id": LIST_ID}}
                },
            }
        }
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code in (201, 202):
            return {"status": "success"}
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/track")
async def track_event(request: TrackRequest):
    url = "https://a.klaviyo.com/api/events/"
    headers = {
        "Authorization": f"Klaviyo-API-Key {KLAVIYO_PRIVATE_KEY}",
        "revision": "2024-10-15",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "data": {
            "type": "event",
            "attributes": {
                "properties": request.properties or {},
                "metric": {
                    "data": {"type": "metric", "attributes": {"name": request.event}}
                },
                "profile": {
                    "data": {"type": "profile", "attributes": {"email": request.email}}
                },
            },
        }
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code in (201, 202):
            return {"status": "event_tracked"}
        raise HTTPException(status_code=response.status_code, detail=response.text)

@modal_app.function(image=env_image, secrets=[modal.Secret.from_name("skinalytics-keys")])
@modal.asgi_app()
def fastapi_app():
    return app