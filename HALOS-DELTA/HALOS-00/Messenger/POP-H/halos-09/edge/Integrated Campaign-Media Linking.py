# halos/campaigns/media_graph.py
from typing import DefaultDict
from collections import defaultdict

class MediaGraph:
    def __init__(self):
        self.campaign_to_media = DefaultDict(list)  # {campaign_id: [media_ids]}
        self.media_to_campaign = {}  # {media_id: campaign_id}
        
    def link_media(self, media_id: str, campaign_id: str):
        """Bi-directional linking"""
        if media_id not in self.media_to_campaign:
            self.campaign_to_media[campaign_id].append(media_id)
            self.media_to_campaign[media_id] = campaign_id
            
    def get_campaign_visuals(self, campaign_id: str) -> List[str]:
        """All media tied to a campaign"""
        return self.campaign_to_media.get(campaign_id, [])
    
    def infer_campaign(self, image: Image) -> Optional[str]:
        """Use CLIP to suggest relevant campaigns"""
        # Implementation would use:
        # model = SentenceTransformer('clip-ViT-B-32')
        # embeddings = model.encode(image)
        # Compare with campaign topic embeddings
        pass