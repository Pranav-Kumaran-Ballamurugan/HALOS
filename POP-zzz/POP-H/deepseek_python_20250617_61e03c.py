import os
import sys
from typing import Dict, Any, Optional
from enum import Enum

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"

class HALOSCore:
    def __init__(self):
        # Existing initialization
        self.llm_router = LLMRouter()
        self.security_lab = EnhancedSecurityLab()
        self.code_doctor = CodeDoctorPro()
        # ... other existing components
        
        # New components
        self.research_agent = ResearchAgent()
        self.vision_module = VisionModule()
        self.memory_system = AdvancedMemorySystem()
        self.finance_tracker = FinanceTrackerPlus()