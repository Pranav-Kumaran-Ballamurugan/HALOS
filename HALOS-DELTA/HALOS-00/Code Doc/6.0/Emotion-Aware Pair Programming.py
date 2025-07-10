class EmotionAwareAI:  
    def __init__(self):  
        self.fer = DeepFace()  # Facial emotion recognition  
        self.voice_analyzer = ToneAnalyzer()  
      
    async def detect_frustration(self, webcam_feed, audio):  
        """Detect developer frustration via:  
        - Facial expression (webcam)  
        - Voice tone (mic)  
        - Typing cadence (keyboard events)"""  
        emotion = self.fer.analyze(webcam_feed)  
        tone = self.voice_analyzer.detect_stress(audio)  
        return emotion["angry"] > 0.8 or tone["frustration"] > 0.7  
      
    async def offer_help(self):  
        """Proactive support"""  
        if await self.detect_frustration():  
            self.voice_engine.speak(  
                "I notice you're stuck. Would you like me to: "  
                "1. Explain this error 2. Suggest fixes 3. Call a human?"  
            )  