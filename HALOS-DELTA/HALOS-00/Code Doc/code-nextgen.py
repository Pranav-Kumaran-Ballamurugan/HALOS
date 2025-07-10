import difflib
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import torch
from fastapi import BackgroundTasks
from pymongo import MongoClient
from unidiff import PatchSet

# GPU-accelerated components
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
gpt_queue = ThreadPoolExecutor(max_workers=4)  # GPU-concurrent

class CodeDoctorEnhanced:
    def __init__(self):
        self.db = MongoClient(os.getenv("MONGO_URI")).code_doctor
        self.embedding_cache = {}

    async def analyze_with_ai(self, code: str, context: dict) -> dict:
        """Enhanced AI analysis with semantic search"""
        # Get similar historical issues
        similar = self.find_similar_code_issues(code)
        
        # Build GPT prompt with context
        prompt = self._build_refactor_prompt(code, context, similar)
        
        # Submit to GPU queue
        future = gpt_queue.submit(
            self._get_gpt_response,
            prompt,
            temperature=0.3
        )
        return await self._process_ai_response(future.result(), code)

    def generate_unidiff(self, old_code: str, new_code: str) -> dict:
        """Generate unified diff with patch metadata"""
        diff = difflib.unified_diff(
            old_code.splitlines(),
            new_code.splitlines(),
            lineterm=''
        )
        diff_text = '\n'.join(diff)
        patch = PatchSet(diff_text)
        
        return {
            "diff": diff_text,
            "metadata": [{
                "file": p.target_file,
                "additions": p.added,
                "deletions": p.removed,
                "hunks": [{
                    "start": h.target_start,
                    "length": h.target_length,
                    "changes": len(h)
                } for h in p]
            } for p in patch]
        }

    def find_similar_code_issues(self, code: str) -> list:
        """Semantic search for similar historical issues"""
        embedding = self._get_code_embedding(code)
        results = self.db.issues.aggregate([{
            "$vectorSearch": {
                "queryVector": embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": 5,
                "index": "semantic_search"
            }
        }])
        return list(results)

    def _get_code_embedding(self, code: str) -> list:
        """Get embedding vector for code (cached)"""
        if code not in self.embedding_cache:
            self.embedding_cache[code] = embedding_model.encode(
                code,
                convert_to_tensor=False
            ).tolist()
        return self.embedding_cache[code]

    def _build_refactor_prompt(self, code: str, context: dict, similar: list) -> str:
        """Construct AI prompt with context"""
        examples = "\n".join([f"// Similar issue {i}:\n{s['code']}" 
                           for i, s in enumerate(similar[:3])])
        
        return f"""
        Refactor this {context.get('language', 'Python')} code considering:
        - Modularity improvements
        - Performance optimizations
        - Microservice decomposition opportunities
        - Security hardening
        
        Code:
        ```{code}```
        
        Historical similar issues:
        {examples}
        
        Suggest specific refactors with:
        1. Priority level (1-5)
        2. Change description
        3. Unified diff format changes
        """

# FastAPI Endpoints
@app.post("/refactor")
async def refactor_code(request: CodeAnalysisRequest, background_tasks: BackgroundTasks):
    """AI-powered refactoring endpoint"""
    doctor = CodeDoctorEnhanced()
    analysis = await doctor.analyze_with_ai(request.code, {
        "language": request.language,
        "user": request.user
    })
    
    # Store in background for semantic search
    background_tasks.add_task(store_analysis_results, analysis)
    return analysis

@app.post("/messenger/webhook")
async def handle_messenger_webhook(message: dict):
    """Process code snippets from HALOS messenger"""
    if message['type'] == 'code_submission':
        doctor = CodeDoctorEnhanced()
        analysis = await doctor.analyze_with_ai(
            message['code'],
            {"source": "messenger", "campaign": message.get('campaign')}
        )
        
        # Send back to messenger
        await halos_messenger.send(
            message['channel'],
            f"Code analysis complete: {analysis['summary']}"
        )
        
        return {"status": "processed"}
    return {"status": "ignored"}

@app.post("/batch-analyze")
async def batch_analyze(files: list[UploadFile]):
    """GPU-accelerated batch processing"""
    doctor = CodeDoctorEnhanced()
    results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            code = await file.read()
            futures.append(executor.submit(
                doctor.analyze_with_ai,
                code.decode(),
                {"filename": file.filename}
            ))
        
        for future in futures:
            results.append(future.result())
    
    return {"results": results}

# MongoDB Schema for Code Embeddings
"""
db.issues.createIndex({
    "embedding": "cosmosSearch"
}, {
    "cosmosSearchOptions": {
        "kind": "vector-ivf",
        "numLists": 1,
        "similarity": "COS",
        "dimensions": 384
    }
})
"""