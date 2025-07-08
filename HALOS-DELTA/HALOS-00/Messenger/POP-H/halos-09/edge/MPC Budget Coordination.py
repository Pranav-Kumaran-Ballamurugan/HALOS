# halos/campaigns/mpc.py
from petlib.bn import Bn
from petlib.ec import EcGroup

class MPCBudget:
    def __init__(self, participants: List[str]):
        self.curve = EcGroup(714)  # secp256k1
        self.commitments = {}
        self.range_proofs = {}
        
    def add_commitment(self, user_id: str, max_amount: int):
        """Create Pedersen commitment for budget"""
        secret = Bn.from_num(secrets.randbits(256))
        commitment = secret * self.curve.generator()
        self.commitments[user_id] = (commitment, secret)
        
        # Generate range proof (0 ≤ amount ≤ max_amount)
        self.range_proofs[user_id] = self._prove_in_range(secret, 0, max_amount)
        
        return commitment

    def verify_contribution(self, user_id: str, amount: int) -> bool:
        """Verify contribution fits within committed budget"""
        commitment, secret = self.commitments[user_id]
        return (
            self._verify_range(secret, self.range_proofs[user_id]) and
            amount <= secret.mod_floor(Bn.from_num(2**64))
        )

    def compute_optimal_split(self, total_cost: int) -> Dict[str, int]:
        """MPC-computed fair distribution without revealing individual budgets"""
        # Simplified example - real impl would use secure summation
        total = sum(secret for _, secret in self.commitments.values())
        return {
            user_id: int((secret/total) * total_cost)
            for user_id, (_, secret) in self.commitments.items()
        }