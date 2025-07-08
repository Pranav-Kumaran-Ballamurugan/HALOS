# halos/sync/merkle.py
import hashlib
from dataclasses import dataclass
from typing import List

@dataclass
class MerkleNode:
    hash: str
    left: 'MerkleNode' = None
    right: 'MerkleNode' = None

class SyncEngine:
    def __init__(self):
        self.roots = {}  # {device_id: MerkleNode}
        
    def _build_tree(self, messages: List[str]) -> MerkleNode:
        """Build Merkle tree from message hashes"""
        leaf_nodes = [
            MerkleNode(hashlib.sha256(msg.encode()).hexdigest())
            for msg in messages
        ]
        
        while len(leaf_nodes) > 1:
            new_level = []
            for i in range(0, len(leaf_nodes), 2):
                left = leaf_nodes[i]
                right = leaf_nodes[i+1] if i+1 < len(leaf_nodes) else left
                combined = left.hash + right.hash
                new_level.append(
                    MerkleNode(hashlib.sha256(combined.encode()).hexdigest(), left, right)
            leaf_nodes = new_level
            
        return leaf_nodes[0]

    def verify_sync(self, device_id: str, messages: List[str]) -> bool:
        """Verify messages match the root hash from another device"""
        their_root = self.roots.get(device_id)
        if not their_root:
            return False
            
        our_root = self._build_tree(messages)
        return our_root.hash == their_root.hash

    def update_root(self, device_id: str, messages: List[str]):
        """Update our root hash for a device"""
        self.roots[device_id] = self._build_tree(messages)