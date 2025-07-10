from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper  # OpenAI's STT
import openai  # GPT integration
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

app = FastAPI(title="HALOS Code Doctor Microservice")

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeAnalysisRequest(BaseModel):
    code: Optional[str] = None
    file_url: Optional[str] = None
    language: str = "python"
    auto_fix: bool = False
    generate_tests: bool = False

class VoiceCommandRequest(BaseModel):
    audio: UploadFile = File(...)
    user_id: str  # For personalized GPT context

# Initialize services
stt_model = whisper.load_model("base")  # STT
code_doctor = CodeDoctorPro()  # Previous implementation
gpt_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/checkcode")
async def analyze_code(request: CodeAnalysisRequest):
    """Main analysis endpoint"""
    if request.file_url:
        code = download_file(request.file_url)
    else:
        code = request.code
    
    # Core analysis
    analysis = code_doctor.analyze_security(code, request.language)
    test_cases = code_doctor.generate_tests(code, request.language) if request.generate_tests else None
    
    # AI-enhanced suggestions
    ai_analysis = await get_ai_analysis(code, request.language)
    
    return {
        "security": analysis,
        "tests": test_cases,
        "ai_suggestions": ai_analysis,
        "auto_fix_available": request.auto_fix
    }

@app.post("/secure")
async def harden_code(request: CodeAnalysisRequest):
    """Security-specific hardening"""
    analysis = await analyze_code(request)
    
    # Apply security-specific transformations
    hardened_code, patches = apply_security_patches(
        request.code, 
        analysis['security']['security_issues']
    )
    
    return {
        "hardened_code": hardened_code,
        "applied_patches": patches,
        "remaining_issues": analysis['security']
    }

@app.post("/autofix")
async def auto_fix_code(request: CodeAnalysisRequest):
    """AI-powered automatic fixing"""
    analysis = await analyze_code(request)
    
    # Get AI-generated fixes
    fixes = []
    for issue in analysis['security']['security_issues']:
        fix_prompt = f"""
        Vulnerability: {issue['description']}
        Code Context: ```{request.code}```
        Generate a secure replacement just for the vulnerable part.
        """
        
        response = gpt_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": fix_prompt}],
            temperature=0.3
        )
        
        fixes.append({
            "issue": issue,
            "ai_fix": response.choices[0].message.content
        })
    
    return {"fixes": fixes}

@app.post("/voice-command")
async def handle_voice_command(request: VoiceCommandRequest):
    """Process voice commands like 'Check this contract'"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await request.audio.read())
        tmp_path = tmp.name
    
    # Speech-to-text
    result = stt_model.transcribe(tmp_path)
    os.unlink(tmp_path)
    command = result["text"].lower()
    
    # Command parsing
    if "check" in command and "contract" in command:
        action = "analyze solidity"
    elif "fix" in command and "vulnerability" in command:
        action = "autofix"
    else:
        action = "analyze"
    
    return {"command": command, "action": action}

async def get_ai_analysis(code: str, language: str) -> dict:
    """Get AI-powered code review"""
    prompt = f"""
    Perform code review for this {language} code:
    ```{code}```
    
    Analyze for:
    1. Security vulnerabilities
    2. Performance optimizations
    3. Code smells
    4. Best practice violations
    
    Format findings as:
    - [Category] Concern (Confidence %)
      • Suggestion
      • Reference
    """
    
    response = gpt_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return parse_ai_response(response.choices[0].message.content)

def parse_ai_response(text: str) -> dict:
    """Structure AI response into categories"""
    # Implementation would parse the bullet points into structured data
    return {"analysis": text}  # Simplified for demo

def apply_security_patches(code: str, issues: list) -> tuple:
    """Apply security patches based on analysis"""
    # Implementation would modify the code
    return code + "\n# Security patches applied", []