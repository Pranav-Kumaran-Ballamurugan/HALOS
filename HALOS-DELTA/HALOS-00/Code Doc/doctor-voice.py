from openai import AsyncOpenAI
from PIL import Image
import numpy as np
import whisper
import pygame  # For voice feedback

class CodeDoctorMultimodal:
    def __init__(self):
        # **1. Voice & Image Models**
        self.stt = whisper.load_model("large-v3")  # Speech-to-text
        self.clip = AsyncOpenAI(api_key="sk-...").clip  # Image understanding
        
        # **2. Creative AI**
        self.dalle = AsyncOpenAI().images
        self.codellama = AsyncOpenAI(model="codellama-70b")
        
        # **3. Real-Time UI**
        self.jupyter = JupyterClient()  # For live coding
        
    async def voice_command(self, audio_path: str) -> str:
        """🎤 Convert speech → code changes"""
        text = self.stt.transcribe(audio_path)["text"]
        
        if "fix" in text.lower():
            return await self.analyze_and_fix(text)
        elif "explain" in text.lower():
            return await self.explain_code(text)
        else:
            return "🔊 Try: 'Fix this function' or 'Explain this error'"

    async def image_to_code(self, image: Image) -> dict:
        """📸 Screenshot → Working Code"""
        # Step 1: Describe the image
        description = await self.clip.describe(
            image=np.array(image),
            prompt="What code is in this image?"
        )
        
        # Step 2: Generate runnable code
        response = await self.codellama.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Convert to code: {description}"
            }]
        )
        
        return {
            "code": response.choices[0].message.content,
            "confidence": 0.9  # Mock value
        }

    async def creative_refactor(self, code: str, style: str) -> str:
        """🎨 Make code 'elegant' or 'minimal'"""
        prompt = f"""
        Rewrite this code in a {style} style:
        ```python
        {code}
        ```
        """
        response = await self.codellama.chat.completions.create(
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def live_coding_session(self, websocket: WebSocket):
        """🤖 Real-Time AI Pair Programming"""
        await websocket.accept()
        while True:
            code = await websocket.receive_text()
            
            # Get AI suggestions
            analysis = await self.analyze(code)
            fixes = await self.refactor(code, analysis['suggestions'])
            
            # Send back to IDE
            await websocket.send_json({
                "updated_code": fixes["refactored_code"],
                "voice_feedback": self.text_to_speech(fixes["summary"])
            })

    def text_to_speech(self, text: str) -> str:
        """🔊 Convert text → spoken audio (MP3)"""
        response = AsyncOpenAI().audio.speech.create(
            model="tts-1",
            voice="echo",
            input=text
        )
        return response.content  # Binary MP3