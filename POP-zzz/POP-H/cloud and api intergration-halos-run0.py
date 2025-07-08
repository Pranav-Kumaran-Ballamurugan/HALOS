class HALOSServer:
    def __init__(self, core: HALOSCore):
        self.core = core
        self.app = FastAPI()
        self._setup_routes()
        self.firebase = FirebaseIntegration()

    def _setup_routes(self):
        @self.app.post("/api/llm")
        async def llm_endpoint(request: Request):
            data = await request.json()
            return self.core.llm_router.get_response(data['prompt'], data.get('task_type'))
        
        # ... other API endpoints

    def sync_to_cloud(self, data: Dict):
        self.firebase.push_data("halos_state", data)