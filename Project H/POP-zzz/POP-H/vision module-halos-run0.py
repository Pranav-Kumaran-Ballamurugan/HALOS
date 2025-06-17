class VisionModule:
    def __init__(self):
        self.camera = WebcamIntegration()
        self.document_processor = DocumentProcessor()

    def analyze_image(self, image_source: str) -> Dict:
        if image_source.startswith('cam:'):
            image = self.camera.capture()
        else:
            image = self._load_image(image_source)
        
        return {
            "objects": self._detect_objects(image),
            "text": self._extract_text(image),
            "analysis": self._gpt4_vision_analysis(image)
        }

    def process_document(self, file_path: str) -> Dict:
        if file_path.endswith('.pdf'):
            return self.document_processor.extract_pdf(file_path)
        elif file_path.endswith(('.pptx', '.ppt')):
            return self.document_processor.extract_pptx(file_path)