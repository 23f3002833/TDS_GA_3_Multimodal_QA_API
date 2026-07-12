import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types  # Required for clean raw byte encoding
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 1. Enable CORS (Crucial for the grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin (like the Cloudflare Worker grader)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Define the exact Request Format required by the assignment
class QAInput(BaseModel):
    image_base64: str
    question: str

# 3. Initialize Gemini Client
client = genai.Client()

@app.post("/answer-image")
async def answer_image(data: QAInput):
    # Step A: Clean and decode base64 into raw bytes
    try:
        base64_str = data.image_base64.strip()
        
        # Strip away any header data URIs if the grader prefixes them
        if base64_str.startswith("data:"):
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_str)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Base64 Decoding Failed: {e}\n")
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {str(e)}")

    # Step B: Structuring prompt constraints
    prompt_instructions = (
        f"Analyze the attached document/chart image and answer this exact question: '{data.question}'. "
        "CRITICAL ASSIGNMENT CONSTRAINTS:\n"
        "1. Give only the exact direct answer value.\n"
        "2. If the answer is a number, return ONLY the raw numeric digits/decimals. "
        "Do NOT include any currency symbols ($), commas, units (kg, m, months), text, or explanation.\n"
        "Example: Output '4089.35', NOT '$4,089.35'."
    )

    # Step C: Execute Gemini Multimodal Pipeline
    try:
        # Wrap raw image bytes cleanly using types.Part.from_bytes
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )
        
        # CHANGED: Updated to the current stable 'gemini-3.5-flash' model
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=[image_part, prompt_instructions]
        )
        
        # Standardise and strip trailing punctuation/quotes from output string
        clean_answer = response.text.strip().replace('"', '').replace("'", "")
        print(f"[SUCCESS] Q: {data.question} -> A: {clean_answer}")
        
        return {"answer": clean_answer}
        
    except Exception as e:
        # Catches exact SDK/network runtime trace and prints to terminal
        print(f"\n--- GEMINI API ERROR: {e} ---\n")
        raise HTTPException(status_code=500, detail=f"LLM Processing Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
