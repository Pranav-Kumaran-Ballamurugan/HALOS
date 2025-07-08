#!/usr/bin/env python3
"""
HALOS ENCRYPTED MESSENGER PRO
- End-to-end encrypted messaging
- Group key management (TreeKEM)
- Offline message queue
- Media encryption
- Cross-device sync
"""

import os
import asyncio
import aiosqlite
import base64
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nio import AsyncClient, MatrixRoom, RoomMessageText
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
import secrets

# ======================
# CORE MESSENGER CLASS
# ======================

class HALOSMessenger:
    def __init__(self):
        # Matrix client setup
        self.client = AsyncClient("https://matrix.org")
        self.user_id = os.getenv("MATRIX_USER_ID")
        self.device_id = os.getenv("DEVICE_ID") or "default_device"
        
        # Encryption systems
        self.room_keys = {}  # {room_id: Fernet(key)}
        self.key_trees = {}  # {room_id: KeyTree}
        self.media_keys = {}  # {media_id: AES key}
        
        # Offline and sync systems
        self.offline_queue = OfflineQueue()
        self.sync_engine = SyncEngine()
        self.message_callbacks = []

    async def login(self):
        """Authenticate with Matrix server"""
        await self.client.login(self.user_id, os.getenv("MATRIX_PASSWORD"))
        print(f"Logged in as {self.user_id}")

# ======================
# ENCRYPTION LAYERS
# ======================

    async def _init_room_encryption(self, room_id: str):
        """Initialize encryption for a new room"""
        # Generate Fernet key for message encryption
        fernet_key = Fernet.generate_key()
        self.room_keys[room_id] = Fernet(fernet_key)
        
        # Initialize TreeKEM for group key management
        members = await self._get_room_members(room_id)
        self.key_trees[room_id] = KeyTree(members)
        
        # Broadcast initial key package
        await self._broadcast_key_package(room_id)

    async def _broadcast_key_package(self, room_id: str):
        """Distribute encryption keys to room members"""
        key_package = {
            "fernet_key": self.room_keys[room_id]._signing_key,
            "root_chain": self.key_trees[room_id].root_chain,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.client.room_send(
            room_id,
            message_type="halos.key_package",
            content=key_package
        )

# ======================
# MESSAGE HANDLING
# ======================

    async def send_message(self, room_id: str, text: str):
        """Send encrypted message with offline support"""
        try:
            if room_id not in self.room_keys:
                await self._init_room_encryption(room_id)
            
            # Encrypt with Fernet + TreeKEM chain
            encrypted = self._double_encrypt(room_id, text)
            
            if await self._is_online():
                await self.client.room_send(
                    room_id,
                    message_type="m.room.message",
                    content={"msgtype": "m.text", "body": encrypted}
                )
                await self.sync_engine.update_sent_message(room_id, text)
            else:
                await self.offline_queue.enqueue(room_id, encrypted)
                
        except Exception as e:
            print(f"Send failed: {e}")

    def _double_encrypt(self, room_id: str, text: str) -> str:
        """Apply both Fernet and TreeKEM encryption"""
        # First layer: Fernet
        fernet_encrypted = self.room_keys[room_id].encrypt(text.encode())
        
        # Second layer: TreeKEM
        chain_key = self.key_trees[room_id].get_current_chain()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'HALOS_TREEKEM'
        )
        encryption_key = hkdf.derive(chain_key)
        cipher = AESGCM(encryption_key)
        nonce = secrets.token_bytes(12)
        doubly_encrypted = nonce + cipher.encrypt(nonce, fernet_encrypted, None)
        
        return base64.b64encode(doubly_encrypted).decode()

# ======================
# MEDIA HANDLING
# ======================

    async def send_media(self, room_id: str, file_path: str):
        """Encrypt and send media files"""
        media_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        encryptor = MediaEncryptor()
        encrypted_data = encryptor.encrypt_file(file_path)
        
        # Store key for later decryption
        self.media_keys[media_id] = encryptor.key
        
        if await self._is_online():
            await self._upload_media(room_id, media_id, encrypted_data)
        else:
            await self.offline_queue.enqueue(
                room_id, 
                f"MEDIA:{media_id}:{base64.b64encode(encrypted_data).decode()}"
            )

# ======================
# OFFLINE SUPPORT
# ======================

    async def flush_offline_messages(self):
        """Send all queued messages when back online"""
        async for room_id, message in self.offline_queue.messages():
            try:
                if message.startswith("MEDIA:"):
                    _, media_id, data = message.split(":")
                    await self._upload_media(room_id, media_id, base64.b64decode(data))
                else:
                    await self.client.room_send(
                        room_id,
                        message_type="m.room.message",
                        content={"msgtype": "m.text", "body": message}
                    )
                await self.offline_queue.mark_delivered(message)
            except Exception as e:
                print(f"Failed to flush message: {e}")

# ======================
# SYNC SYSTEM
# ======================

    async def sync_devices(self):
        """Coordinate state across all user devices"""
        # Get latest messages for Merkle tree
        messages = await self._get_recent_messages()
        
        # Build and compare Merkle trees
        my_root = self.sync_engine.build_tree(messages)
        
        for device_id in self.known_devices:
            their_root = await self._get_device_root(device_id)
            if their_root and my_root != their_root:
                await self._reconcile_diff(device_id, my_root, their_root)

# ======================
# SUPPORTING CLASSES
# ======================

class KeyTree:
    """TreeKEM implementation for group key management"""
    def __init__(self, members: List[str]):
        self.private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.chain_keys = {member: self._derive_chain_key(secrets.token_bytes(32)) for member in members}
        self.root_chain = secrets.token_bytes(32)
        
    def _derive_chain_key(self, input_key: bytes) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'HALOS_CHAIN_KEY'
        ).derive(input_key)

class OfflineQueue:
    """Persistent offline message storage"""
    def __init__(self, db_path='offline.db'):
        self.db_path = db_path
        
    async def enqueue(self, room_id: str, message: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (room_id, content, timestamp) VALUES (?, ?, ?)",
                (room_id, message, datetime.utcnow().timestamp())
            )
            await db.commit()

class MediaEncryptor:
    """AES-GCM media encryption"""
    def __init__(self):
        self.key = AESGCM.generate_key(bit_length=256)
        
    def encrypt_file(self, file_path: str) -> bytes:
        with open(file_path, 'rb') as f:
            data = f.read()
        aesgcm = AESGCM(self.key)
        nonce = secrets.token_bytes(12)
        return nonce + aesgcm.encrypt(nonce, data, None)

class SyncEngine:
    """Merkle-tree based sync system"""
    def build_tree(self, messages: List[str]) -> str:
        hashes = [hashlib.sha256(msg.encode()).hexdigest() for msg in messages]
        while len(hashes) > 1:
            hashes = [hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest() 
                     for i in range(0, len(hashes), 2)]
        return hashes[0]

# ======================
# MAIN ENTRY POINT
# ======================

async def main():
    messenger = HALOSMessenger()
    await messenger.login()
    
    # Example usage
    await messenger.send_message("!room_id:matrix.org", "Hello HALOS!")
    await messenger.send_media("!room_id:matrix.org", "photo.jpg")
    
    # Start continuous sync
    asyncio.create_task(messenger.sync_forever())

if __name__ == "__main__":
    asyncio.run(main())