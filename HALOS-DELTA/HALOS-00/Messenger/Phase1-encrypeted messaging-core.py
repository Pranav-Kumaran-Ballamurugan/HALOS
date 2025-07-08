# halos/messenger/core.py
from matrix_client.api import MatrixHttpApi
from cryptography.fernet import Fernet

class HALOSMessenger:
    def __init__(self, user_id):
        self.client = MatrixHttpApi("https://matrix.org", token=os.getenv("MATRIX_TOKEN"))
        self.room_key_cache = {}  # {room_id: Fernet(key)}
        
    async def _encrypt_for_room(self, room_id: str, text: str) -> str:
        """End-to-end encryption per room"""
        if room_id not in self.room_key_cache:
            key = Fernet.generate_key()
            await self.client.send_state_event(
                room_id, 
                "halos.encryption_key", 
                {"key": key.decode()}
            )
            self.room_key_cache[room_id] = Fernet(key)
        return self.room_key_cache[room_id].encrypt(text.encode()).decode()

    async def send(self, room_id: str, text: str):
        encrypted = await self._encrypt_for_room(room_id, text)
        await self.client.send_message(room_id, encrypted)