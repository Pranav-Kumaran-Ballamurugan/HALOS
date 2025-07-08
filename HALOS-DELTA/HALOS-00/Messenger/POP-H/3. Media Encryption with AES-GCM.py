# halos/encryption/media.py
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64

class MediaEncryptor:
    def __init__(self):
        self.key = AESGCM.generate_key(bit_length=256)
        
    def encrypt_file(self, file_path: str) -> tuple[bytes, bytes]:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        aesgcm = AESGCM(self.key)
        nonce = secrets.token_bytes(12)
        encrypted = aesgcm.encrypt(nonce, data, None)
        return nonce + encrypted
        
    def decrypt_file(self, encrypted_data: bytes, output_path: str):
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        aesgcm = AESGCM(self.key)
        decrypted = aesgcm.decrypt(nonce, ciphertext, None)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)