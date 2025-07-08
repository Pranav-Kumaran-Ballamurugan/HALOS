# halos/encryption/treekem.py
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.asymmetric import x25519
import secrets

class KeyTree:
    def __init__(self, members):
        self.private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        self.chain_keys = {}  # {user_id: bytes}
        self.root_chain = secrets.token_bytes(32)
        
        # Initialize with members
        for member in members:
            self.add_member(member)

    def derive_chain_key(self, input_key: bytes) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'HALOS_CHAIN_KEY'
        ).derive(input_key)

    def add_member(self, user_id: str):
        # Ratchet forward the chain
        new_key = self.derive_chain_key(self.root_chain)
        self.chain_keys[user_id] = new_key
        self.root_chain = self.derive_chain_key(new_key)

    def remove_member(self, user_id: str):
        if user_id in self.chain_keys:
            del self.chain_keys[user_id]
            # Re-key the entire group
            self.root_chain = secrets.token_bytes(32)
            for member in self.chain_keys:
                self.chain_keys[member] = self.derive_chain_key(self.root_chain)