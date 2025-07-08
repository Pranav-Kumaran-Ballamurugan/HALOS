# halos/social/gamification.py
import numpy as np

class ContributionRings:
    def __init__(self):
        self.user_scores = {}  # {user_id: {metric: value}}
        self.ring_metrics = [
            "campaign_contributions",
            "media_shares",
            "planning_sessions"
        ]
        
    def update_score(self, user_id: str, metric: str, value: float):
        """Update gamification metrics"""
        if user_id not in self.user_scores:
            self.user_scores[user_id] = {m: 0 for m in self.ring_metrics}
            
        self.user_scores[user_id][metric] += value
        
    def get_ring_completion(self, user_id: str) -> float:
        """Calculate ring fill percentage (0-1)"""
        if user_id not in self.user_scores:
            return 0
            
        # Normalize across metrics
        scores = [self.user_scores[user_id][m] for m in self.ring_metrics]
        return np.tanh(sum(scores) / len(self.ring_metrics))  # Squash to 0-1