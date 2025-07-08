async def verify_fingerprint(room_id, expected_fingerprint):
    key = self.room_keys[room_id]._signing_key
    actual = hashlib.sha256(key).hexdigest()[:8]
    if actual != expected_fingerprint:
        raise SecurityWarning("Key mismatch!")