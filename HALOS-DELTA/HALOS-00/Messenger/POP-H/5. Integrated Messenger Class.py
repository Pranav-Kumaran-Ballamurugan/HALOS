# halos/messenger/full.py
class HALOSMessengerPro(HALOSMessenger):
    def __init__(self):
        super().__init__()
        self.key_trees = {}  # {room_id: KeyTree}
        self.media_encryptor = MediaEncryptor()
        self.offline_queue = OfflineQueue()
        self.sync_engine = SyncEngine()
        
    async def send_media(self, room_id: str, file_path: str):
        encrypted = self.media_encryptor.encrypt_file(file_path)
        await self.send_message(room_id, base64.b64encode(encrypted).decode())
        
    async def handle_rekey(self, room_id: str):
        """Periodic key rotation"""
        tree = self.key_trees[room_id]
        new_root = secrets.token_bytes(32)
        await self._broadcast_new_root(room_id, new_root)
        
    async def sync_devices(self):
        """Push state to other devices"""
        messages = await self._get_recent_messages()
        self.sync_engine.update_root("this_device", messages)
        
        for device in self.known_devices:
            if self.sync_engine.verify_sync(device, messages):
                await self._push_messages_to_device(device)