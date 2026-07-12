import base64
import os
import http.client
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 1. Enable CORS (Crucial for the grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define Request Format
class QAInput(BaseModel):
    image_base64: str
    question: str

# 3. Read the token from Environment Variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "YOUR_GITHUB_TOKEN_HERE")

@app.post("/answer-image")
async def answer_image(data: QAInput):
    try:
        base64_str = data.image_base64.strip()
        
        # Clean data URI prefix if it exists
        if base64_str.startswith("data:"):
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
        
        # Format the data URI standard string for the model payload
        data_url = f"data:image/png;base64,{base64_str}"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 preparation error: {str(e)}")

    prompt_instructions = (
        f"Analyze the attached image and answer this exact question: '{data.question}'. "
        "CRITICAL RULES:\n"
        "1. Give only the exact direct answer value.\n"
        "2. If the answer is a number, return ONLY the raw numeric digits/decimals. "
        "Do NOT include any currency symbols ($), commas, units (kg, m), or explanation.\n"
        "Example: Output '4089.35', NOT '$4,089.35'."
    )

    # 4. Formulate the raw HTTP request payload for gpt-4o-mini
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
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
        "temperature": 0.1
    }

    try:
        # Standard HTTP client to query the GitHub Models marketplace endpoint
        conn = http.client.HTTPSConnection("://azure.com")
        conn.request("POST", "/chat/completions", json.dumps(payload), headers)
        res = conn.getresponse()
        response_data = json.loads(res.read().decode("utf-8"))
        conn.close()
        
        # Extract the string content answer
        raw_answer = response_data["choices"][0]["message"]["content"]
        clean_answer = raw_answer.strip().replace('"', '').replace("'", "")
        
        print(f"[SUCCESS] Q: {data.question} -> A: {clean_answer}")
        return {"answer": clean_answer}
        
    except Exception as e:
        print(f"[ERROR] Inference Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model execution error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
