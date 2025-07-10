class CodeArtist:  
    def generate_art(self, code: str, style: str) -> Image:  
        """Turn code into visual art"""  
        prompt = f"Code artwork in {style} style:\n```{code}```"  
        return OpenAIClient().images.generate(  
            model="dall-e-3",  
            prompt=prompt,  
            size="1792x1024"  
        )  