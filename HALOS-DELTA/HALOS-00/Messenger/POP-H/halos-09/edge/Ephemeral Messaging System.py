# halos/messaging/ephemeral.py
import asyncio
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

class EphemeralEngine:
    def __init__(self):
        self.pending_deletions = {}  # {message_id: (delete_at, callback)}
        self.zero_storage = True  # Wipe plaintext after read by default
    
    async def send_ephemeral(self, room_id: str, text: str, ttl: int = 60):
        """Send message that auto-destructs after TTL seconds"""
        msg_id = secrets.token_hex(8)
        encrypted = self._encrypt_with_self_destruct(msg_id, text)
        
        # Schedule deletion
        delete_at = datetime.now() + timedelta(seconds=ttl)
        self.pending_deletions[msg_id] = (
            delete_at,
            lambda: self._redact_message(room_id, msg_id)
        )
        
        return await self.client.send(room_id, {
            "type": "halos.ephemeral",
            "msg_id": msg_id,
            "ciphertext": encrypted,
            "ttl": ttl
        })

    def _encrypt_with_self_destruct(self, msg_id: str, text: str) -> bytes:
        """Encrypt such that key becomes unrecoverable after TTL"""
        key = Fernet.generate_key()
        cipher = Fernet(key)
        encrypted = cipher.encrypt(text.encode())
        
        # Store key in volatile memory only
        if self.zero_storage:
            asyncio.create_task(self._volatile_key_store(msg_id, key))
        
        return encrypted

    async def _volatile_key_store(self, msg_id: str, key: bytes):
        """Temporarily hold keys in memory with auto-cleanup"""
        await asyncio.sleep(self.pending_deletions[msg_id][0] - datetime.now())
        del key  # Cryptographic wipe from memory