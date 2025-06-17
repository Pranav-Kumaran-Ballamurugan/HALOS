class AdvancedMemorySystem:
    def __init__(self):
        self.vector_db = ChromaDB()
        self.embedder = SentenceTransformer()
        self.file_storage = FileStorage()

    def store_memory(self, content: str, tags: List[str] = None, files: List[str] = None) -> str:
        embedding = self.embedder.encode(content)
        memory_id = self.vector_db.store(embedding, content)
        
        if tags:
            self._auto_tag(memory_id, tags)
        
        if files:
            self.file_storage.attach_files(memory_id, files)
        
        return memory_id

    def search_memories(self, query: str) -> List[Dict]:
        query_embedding = self.embedder.encode(query)
        results = self.vector_db.search(query_embedding)
        return [{
            "content": r['content'],
            "tags": self._get_tags(r['id']),
            "files": self.file_storage.get_attachments(r['id'])
        } for r in results]