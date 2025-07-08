# halos/messenger/core.py
import os
import asyncio
from nio import AsyncClient, MatrixRoom, RoomMessageText
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

class HALOSMessenger:
    def __init__(self):
        self.user_id = os.getenv("MATRIX_USER_ID")
        self.password = os.getenv("MATRIX_PASSWORD")
        self.homeserver = "https://matrix.org"
        self.client = AsyncClient(self.homeserver)
        self.room_keys = {}  # {room_id: Fernet(key)}
        self.message_callbacks = []

    async def login(self):
        await self.client.login(self.user_id, self.password)
        print(f"Logged in as {self.user_id}")

    async def _generate_room_key(self, room_id: str) -> Fernet:
        """Generate and exchange encryption key for a room"""
        key = Fernet.generate_key()
        await self.client.room_send(
            room_id,
            message_type="halos.encryption_key",
            content={"key": key.decode()}
        )
        return Fernet(key)

    async def join_room(self, room_id: str):
        await self.client.join(room_id)
        if room_id not in self.room_keys:
            self.room_keys[room_id] = await self._generate_room_key(room_id)

    async def send_message(self, room_id: str, text: str):
        """Send encrypted message to a room"""
        if room_id not in self.room_keys:
            await self.join_room(room_id)
        
        encrypted = self.room_keys[room_id].encrypt(text.encode()).decode()
        await self.client.room_send(
            room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": encrypted}
        )

    async def decrypt_message(self, room_id: str, encrypted: str) -> str:
        """Decrypt received message"""
        return self.room_keys[room_id].decrypt(encrypted.encode()).decode()

    async def add_message_handler(self, callback):
        """Add callback for incoming messages"""
        self.message_callbacks.append(callback)

    async def sync_forever(self):
        """Continuous sync loop"""
        while True:
            response = await self.client.sync(30000)
            for room_id, room_info in response.rooms.join.items():
                for event in room_info.timeline.events:
                    if isinstance(event, RoomMessageText):
                        try:
                            decrypted = await self.decrypt_message(room_id, event.body)
                            for callback in self.message_callbacks:
                                await callback(room_id, event.sender, decrypted)
                        except Exception as e:
                            print(f"Decryption failed: {e}")

if __name__ == "__main__":
    messenger = HALOSMessenger()
    
    async def message_received(room_id, sender, text):
        print(f"[{room_id}] {sender}: {text}")
    
    async def run():
        await messenger.login()
        await messenger.add_message_handler(message_received)
        await messenger.join_room("!YourRoomId:matrix.org")
        await messenger.send_message("!YourRoomId:matrix.org", "Hello HALOS!")
        await messenger.sync_forever()
    
    asyncio.get_event_loop().run_until_complete(run())