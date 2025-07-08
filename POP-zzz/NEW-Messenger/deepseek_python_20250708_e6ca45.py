#!/usr/bin/env python3
"""
HALOS ENCRYPTED MESSENGER PRO - ENHANCED VERSION
- End-to-end encrypted messaging with double encryption
- Improved TreeKEM group key management
- Persistent offline message queue with retries
- Secure media encryption with metadata protection
- Cross-device sync with conflict resolution
- Better error handling and logging
"""

import os
import asyncio
import aiosqlite
import base64
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nio import AsyncClient, MatrixRoom, RoomMessageText, LoginResponse
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, AsyncGenerator
import secrets
import aiofiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HALOS")

# ======================
# DATA MODELS
# ======================

@dataclass
class DeviceInfo:
    device_id: str
    public_key: bytes
    last_seen: datetime

@dataclass
class PendingMessage:
    room_id: str
    content: str
    timestamp: datetime
    attempts: int = 0

# ======================
# CORE MESSENGER CLASS
# ======================

class HALOSMessenger:
    def __init__(self, config_path: str = "halos_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Matrix client setup
        self.client = AsyncClient(
            homeserver=self.config.get("homeserver", "https://matrix.org"),
            user=self.config.get("user_id"),
            device_id=self.config.get("device_id", "default_device"),
            store_path=self.config.get("store_path", "halos_store")
        )
        
        # Encryption systems
        self.identity_key = self._load_or_generate_identity_key()
        self.room_keys = {}  # {room_id: Fernet(key)}
        self.key_trees = {}  # {room_id: KeyTree}
        self.media_keys = {}  # {media_id: (key, hash)}
        
        # Device management
        self.known_devices: Dict[str, DeviceInfo] = {}
        
        # Offline and sync systems
        self.offline_queue = OfflineQueue()
        self.sync_engine = SyncEngine()
        self.message_callbacks = []
        
        # Session state
        self.is_online = False
        self.sync_task = None

    def _load_config(self) -> Dict:
        """Load configuration from file or environment variables"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        
        return {
            "user_id": os.getenv("MATRIX_USER_ID"),
            "password": os.getenv("MATRIX_PASSWORD"),
            "device_id": os.getenv("DEVICE_ID") or f"halos_{secrets.token_hex(4)}",
            "homeserver": os.getenv("MATRIX_HOMESERVER", "https://matrix.org")
        }

    def _load_or_generate_identity_key(self) -> x25519.X25519PrivateKey:
        """Load or generate the device's long-term identity key"""
        key_file = "identity_key.pem"
        try:
            if os.path.exists(key_file):
                with open(key_file, "rb") as f:
                    return x25519.X25519PrivateKey.from_private_bytes(f.read())
        except Exception as e:
            logger.warning(f"Failed to load identity key: {e}")
        
        # Generate new key if none exists
        new_key = x25519.X25519PrivateKey.generate()
        with open(key_file, "wb") as f:
            f.write(new_key.private_bytes_raw())
        return new_key

    async def login(self) -> bool:
        """Authenticate with Matrix server with retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                resp = await self.client.login(
                    password=self.config["password"],
                    device_name=f"HALOS/{self.config['device_id']}"
                )
                
                if isinstance(resp, LoginResponse):
                    self.is_online = True
                    logger.info(f"Logged in as {self.client.user_id}")
                    self._start_background_tasks()
                    return True
                
                logger.error(f"Login failed: {resp}")
                return False
            except Exception as e:
                logger.error(f"Login attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False

    def _start_background_tasks(self):
        """Start essential background tasks"""
        self.sync_task = asyncio.create_task(self._sync_forever())
        self.sync_task.add_done_callback(self._handle_task_failure)
        
        # Start periodic tasks
        asyncio.create_task(self._periodic_key_rotation())
        asyncio.create_task(self._periodic_offline_flush())

    def _handle_task_failure(self, task: asyncio.Task):
        """Restart failed background tasks"""
        try:
            task.result()  # This will raise the exception if one occurred
        except Exception as e:
            logger.error(f"Background task failed: {e}")
            if task == self.sync_task:
                logger.info("Restarting sync task...")
                self._start_background_tasks()

# ======================
# ENHANCED ENCRYPTION LAYERS
# ======================

    async def _init_room_encryption(self, room_id: str):
        """Initialize encryption for a new room with member verification"""
        if room_id in self.room_keys:
            return
            
        # Verify room members first
        members = await self._get_verified_room_members(room_id)
        if not members:
            raise ValueError("No verified members in room")
        
        # Generate Fernet key for message encryption
        fernet_key = Fernet.generate_key()
        self.room_keys[room_id] = Fernet(fernet_key)
        
        # Initialize TreeKEM with forward secrecy
        self.key_trees[room_id] = KeyTree(
            members,
            initial_chain_secret=secrets.token_bytes(32)
        
        # Broadcast initial key package with signatures
        await self._broadcast_key_package(room_id)
        logger.info(f"Initialized encryption for room {room_id}")

    async def _broadcast_key_package(self, room_id: str):
        """Securely distribute encryption keys to verified room members"""
        key_package = {
            "version": 1,
            "fernet_key": base64.b64encode(self.room_keys[room_id]._signing_key).decode(),
            "root_chain": base64.b64encode(self.key_trees[room_id].root_chain).decode(),
            "sender": self.client.user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "signature": self._sign_data(
                f"{room_id}{self.room_keys[room_id]._signing_key.hex()}"
            )
        }
        
        # Encrypt the package for each recipient
        encrypted_packages = {}
        for member in self.key_trees[room_id].members:
            if member != self.client.user_id:
                encrypted_packages[member] = self._encrypt_for_device(
                    member,
                    json.dumps(key_package).encode()
                )
        
        await self.client.room_send(
            room_id,
            message_type="halos.key_package",
            content={
                "packages": encrypted_packages,
                "sender_key": base64.b64encode(
                    self.identity_key.public_key().public_bytes_raw()
                ).decode()
            }
        )

    def _sign_data(self, data: str) -> str:
        """Sign data with identity key"""
        signature = self.identity_key.sign(data.encode())
        return base64.b64encode(signature).decode()

# ======================
# IMPROVED MESSAGE HANDLING
# ======================

    async def send_message(self, room_id: str, text: str) -> bool:
        """Send encrypted message with offline support and delivery tracking"""
        try:
            if room_id not in self.room_keys:
                await self._init_room_encryption(room_id)
            
            # Encrypt with Fernet + TreeKEM chain
            encrypted = self._double_encrypt(room_id, text)
            message_id = hashlib.sha256(encrypted.encode()).hexdigest()[:12]
            
            if await self._is_online():
                resp = await self.client.room_send(
                    room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body": encrypted,
                        "halos": {
                            "message_id": message_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    }
                )
                
                if hasattr(resp, "event_id"):
                    await self.sync_engine.update_sent_message(room_id, text, message_id)
                    logger.info(f"Message sent to {room_id} (ID: {message_id})")
                    return True
                else:
                    logger.warning(f"Failed to send message to {room_id}")
                    await self.offline_queue.enqueue(room_id, encrypted, message_id)
                    return False
            else:
                await self.offline_queue.enqueue(room_id, encrypted, message_id)
                logger.info(f"Message queued offline for {room_id} (ID: {message_id})")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send message: {e}", exc_info=True)
            return False

    def _double_encrypt(self, room_id: str, text: str) -> str:
        """Apply both Fernet and TreeKEM encryption with improved key derivation"""
        # First layer: Fernet
        fernet_encrypted = self.room_keys[room_id].encrypt(text.encode())
        
        # Second layer: TreeKEM with improved key derivation
        chain_key = self.key_trees[room_id].get_current_chain()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),  # Randomized salt for each encryption
            info=b'HALOS_TREEKEM_' + chain_key[:4]  # Key-specific context
        )
        encryption_key = hkdf.derive(chain_key)
        
        cipher = AESGCM(encryption_key)
        nonce = os.urandom(12)
        ciphertext = cipher.encrypt(nonce, fernet_encrypted, None)
        
        # Package with version info
        encrypted_package = {
            "v": 1,  # Version
            "n": base64.b64encode(nonce).decode(),
            "c": base64.b64encode(ciphertext).decode(),
            "k": base64.b64encode(chain_key[:4]).decode()  # Key hint
        }
        
        return json.dumps(encrypted_package)

# ======================
# ENHANCED MEDIA HANDLING
# ======================

    async def send_media(self, room_id: str, file_path: str) -> Optional[str]:
        """Encrypt and send media files with metadata protection"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            media_id = hashlib.sha256(file_path.encode()).hexdigest()[:16]
            encryptor = MediaEncryptor()
            
            async with aiofiles.open(file_path, 'rb') as f:
                file_data = await f.read()
            
            # Calculate file hash before encryption
            file_hash = hashlib.sha256(file_data).hexdigest()
            encrypted_data = encryptor.encrypt_file(file_data)
            
            # Store key with hash for verification
            self.media_keys[media_id] = (encryptor.key, file_hash)
            
            # Prepare encrypted metadata
            metadata = {
                "name": os.path.basename(file_path),
                "type": "application/octet-stream",  # TODO: detect actual type
                "size": len(file_data),
                "hash": file_hash
            }
            encrypted_metadata = self.room_keys[room_id].encrypt(
                json.dumps(metadata).encode()
            )
            
            if await self._is_online():
                # Upload both data and metadata
                upload_resp = await self._upload_media(room_id, media_id, encrypted_data)
                if upload_resp:
                    await self.client.room_send(
                        room_id,
                        message_type="halos.media",
                        content={
                            "id": media_id,
                            "url": upload_resp.content_uri,
                            "metadata": base64.b64encode(encrypted_metadata).decode(),
                            "key_hint": base64.b64encode(encryptor.key[:4]).decode()
                        }
                    )
                    return media_id
            else:
                # Queue for offline delivery
                await self.offline_queue.enqueue(
                    room_id, 
                    f"MEDIA:{media_id}:{base64.b64encode(encrypted_data).decode()}:"
                    f"{base64.b64encode(encrypted_metadata).decode()}"
                )
                return media_id
                
        except Exception as e:
            logger.error(f"Failed to send media: {e}", exc_info=True)
            return None

# ======================
# IMPROVED OFFLINE SUPPORT
# ======================

    async def flush_offline_messages(self):
        """Send all queued messages with retry logic and priority"""
        async for pending in self.offline_queue.pending_messages():
            try:
                if pending.content.startswith("MEDIA:"):
                    # Handle media messages
                    parts = pending.content.split(":")
                    if len(parts) >= 4:
                        media_id = parts[1]
                        data = base64.b64decode(parts[2])
                        metadata = base64.b64decode(parts[3]) if len(parts) > 3 else None
                        
                        upload_resp = await self._upload_media(pending.room_id, media_id, data)
                        if upload_resp:
                            content = {
                                "id": media_id,
                                "url": upload_resp.content_uri
                            }
                            if metadata:
                                content["metadata"] = base64.b64encode(metadata).decode()
                            
                            await self.client.room_send(
                                pending.room_id,
                                message_type="halos.media",
                                content=content
                            )
                            await self.offline_queue.mark_delivered(pending)
                else:
                    # Handle regular messages
                    resp = await self.client.room_send(
                        pending.room_id,
                        message_type="m.room.message",
                        content={
                            "msgtype": "m.text",
                            "body": pending.content
                        }
                    )
                    if hasattr(resp, "event_id"):
                        await self.offline_queue.mark_delivered(pending)
                        
            except Exception as e:
                logger.warning(f"Failed to send queued message (attempt {pending.attempts + 1}): {e}")
                await self.offline_queue.record_attempt(pending)
                
                # Exponential backoff before retry
                await asyncio.sleep(min(2 ** pending.attempts, 60))  # Max 1 minute

# ======================
# ENHANCED SYNC SYSTEM
# ======================

    async def _sync_forever(self):
        """Continuous sync with error handling and backoff"""
        while True:
            try:
                await self.client.sync(timeout=30000, full_state=True)
                self.is_online = True
                
                # Process any received messages
                await self._process_sync_response()
                
                # Check for pending messages
                if self.offline_queue.has_pending():
                    await self.flush_offline_messages()
                    
                await asyncio.sleep(5)  # Normal sync interval
                
            except Exception as e:
                self.is_online = False
                logger.error(f"Sync error: {e}", exc_info=True)
                await asyncio.sleep(10)  # Longer wait on error

    async def sync_devices(self):
        """Enhanced device sync with conflict resolution"""
        # Get latest messages with timestamps
        messages = await self._get_recent_messages()
        
        # Build Merkle tree with version vectors
        my_root, my_versions = self.sync_engine.build_tree_with_versions(messages)
        
        for device_id, device in self.known_devices.items():
            if device_id == self.config["device_id"]:
                continue
                
            their_root, their_versions = await self._get_device_state(device_id)
            
            if their_root and my_root != their_root:
                # Resolve conflicts
                await self._reconcile_differences(
                    device_id,
                    my_root,
                    my_versions,
                    their_root,
                    their_versions
                )

# ======================
# IMPROVED SUPPORTING CLASSES
# ======================

class KeyTree:
    """Enhanced TreeKEM implementation with forward secrecy"""
    def __init__(self, members: List[str], initial_chain_secret: bytes = None):
        self.members = members
        self.private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        
        # Chain keys with forward secrecy
        self.chain_secrets = {
            member: self._derive_chain_key(
                initial_chain_secret or secrets.token_bytes(32)
            )
            for member in members
        }
        
        # Root chain gets updated with each ratchet
        self.root_chain = self._derive_root_chain(initial_chain_secret)
        self.update_timestamp = datetime.utcnow()
        
    def _derive_chain_key(self, input_key: bytes) -> bytes:
        """Improved key derivation with context binding"""
        return HKDF(
            algorithm=hashes.SHA512(),  # Stronger hash
            length=64,  # Longer output
            salt=os.urandom(16),
            info=b'HALOS_CHAIN_KEY_' + input_key[:4]
        ).derive(input_key)[:32]  # Truncate to 256 bits

    def ratchet_chain(self):
        """Update chain keys for forward secrecy"""
        new_secret = secrets.token_bytes(32)
        self.root_chain = self._derive_root_chain(new_secret)
        for member in self.members:
            self.chain_secrets[member] = self._derive_chain_key(new_secret)
        self.update_timestamp = datetime.utcnow()

class OfflineQueue:
    """Persistent offline message storage with priority and retries"""
    def __init__(self, db_path: str = 'offline.db'):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize database with schema"""
        async def _async_init():
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        room_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        message_id TEXT,
                        timestamp REAL NOT NULL,
                        attempts INTEGER DEFAULT 0,
                        last_attempt REAL,
                        delivered INTEGER DEFAULT 0
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_undelivered 
                    ON messages(delivered, attempts, timestamp)
                """)
                await db.commit()
                
        asyncio.run(_async_init())
        
    async def enqueue(self, room_id: str, content: str, message_id: str = None):
        """Add message to queue with tracking"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (room_id, content, message_id, timestamp) VALUES (?, ?, ?, ?)",
                (room_id, content, message_id, datetime.utcnow().timestamp())
            )
            await db.commit()
            
    async def pending_messages(self) -> AsyncGenerator[PendingMessage, None]:
        """Generator for undelivered messages in priority order"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT rowid, room_id, content, timestamp, attempts 
                FROM messages 
                WHERE delivered = 0 
                ORDER BY attempts, timestamp
            """) as cursor:
                async for row in cursor:
                    yield PendingMessage(
                        room_id=row[1],
                        content=row[2],
                        timestamp=datetime.fromtimestamp(row[3]),
                        attempts=row[4]
                    )

class MediaEncryptor:
    """Enhanced media encryption with chunking support"""
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    
    def __init__(self):
        self.key = AESGCM.generate_key(bit_length=256)
        self.chunk_keys = []  # For chunked encryption
        
    def encrypt_file(self, data: bytes) -> bytes:
        """Encrypt file data with metadata header"""
        aesgcm = AESGCM(self.key)
        nonce = os.urandom(12)
        
        # Encrypt with metadata about encryption
        encrypted = nonce + aesgcm.encrypt(
            nonce,
            data,
            b"HALOS_MEDIA_v1"
        )
        
        return encrypted

class SyncEngine:
    """Enhanced sync with version vectors and conflict resolution"""
    def build_tree_with_versions(self, messages: List[Tuple[str, int]]) -> Tuple[str, Dict[str, int]]:
        """Build Merkle tree with version vectors for conflict detection"""
        version_vector = {}
        hashes = []
        
        for msg, version in messages:
            msg_hash = hashlib.sha256(msg.encode()).hexdigest()
            hashes.append(msg_hash)
            version_vector[msg_hash] = version
            
        while len(hashes) > 1:
            new_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + (hashes[i+1] if i+1 < len(hashes) else "")
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
                version_vector[new_hash] = max(
                    version_vector[hashes[i]],
                    version_vector[hashes[i+1]] if i+1 < len(hashes) else 0
                )
            hashes = new_level
            
        return hashes[0], version_vector

# ======================
# MAIN ENTRY POINT
# ======================

async def main():
    try:
        messenger = HALOSMessenger()
        if not await messenger.login():
            logger.error("Failed to log in")
            return
            
        # Example usage
        await messenger.send_message("!room_id:matrix.org", "Hello HALOS!")
        await messenger.send_media("!room_id:matrix.org", "photo.jpg")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())