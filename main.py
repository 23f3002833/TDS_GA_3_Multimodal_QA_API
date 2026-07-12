import base64
import os
import http.client
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 1. Enable Global CORS (Crucial for the grader worker engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Match standard API Input Contract schema
class QAInput(BaseModel):
    image_base64: str
    question: str

# 3. Read token securely from environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

@app.post("/answer-image")
async def answer_image(data: QAInput):
    try:
        base64_str = data.image_base64.strip()
        
        # Clean prefix wrappers safely
        if base64_str.startswith("data:"):
            if "," in base64_str:
                # Safely extract everything after the first comma
                base64_str = base64_str.split(",", 1)[1]
        
        # Structure valid image reference standard block
        data_url = f"data:image/png;base64,{base64_str}"
    except Exception as e:
        print(f"[ERROR] Decoding setup failed: {e}")
        raise HTTPException(status_code=400, detail="Malformed base64 image data payload.")

    # Strict operational boundary rules for grading normalization
    prompt_instructions = (
        f"Analyze the attached document/chart image data and answer this question: '{data.question}'. "
        "CRITICAL RULES:\n"
        "1. Give only the exact direct answer value string.\n"
        "2. If the answer is numeric, return ONLY the raw digit formatting value decimals. "
        "Do NOT include any currency symbols ($), commas, units (kg, months), or words.\n"
        "Example: format '4089.35', NOT '$4,089.35'."
    )

    # Prepare standard API context headers
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Structure model context request payloads using gpt-4o-mini
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_instructions},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        "temperature": 0.0
    }

    try:
        # FIXED: Routing via universal proxy adapter to allow clean raw HTTP requests
        conn = http.client.HTTPSConnection("models.inference.ai.azure.com")
        conn.request("POST", "/chat/completions", json.dumps(payload), headers)
        
        # FIXED: Removed the invalid `getcall` assignment typo causing the 500 error
        res = conn.getresponse()
        response_body = res.read().decode("utf-8")
        response_data = json.loads(response_body)
        conn.close()
        
        # Read token structure elements safely
        if "choices" in response_data:
            raw_answer = response_data["choices"][0]["message"]["content"]
            clean_answer = raw_answer.strip().replace('"', '').replace("'", "")
            print(f"[SUCCESS] Q: {data.question} -> A: {clean_answer}")
            return {"answer": clean_answer}
        else:
            print(f"[CRITICAL] Unexpected response mapping schema: {response_data}")
            raise HTTPException(status_code=502, detail="Upstream inference response extraction mismatch.")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Target runtime execution crashed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal proxy server engine error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
