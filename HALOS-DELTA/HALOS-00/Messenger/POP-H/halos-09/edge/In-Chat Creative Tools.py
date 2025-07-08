# halos/features/remix.py
import openai
from PIL import Image
import io

class CreativeRemix:
    def __init__(self):
        self.prompt_cache = {}
        
    async def remix_from_chat(self, messages: List[str], style: str = "kpop-poster"):
        """Generate creative content from chat history"""
        prompt = f"""
        Create a {style} based on these messages:
        {chr(10).join(messages)}
        
        Guidelines:
        - Vibrant K-pop aesthetic
        - Include key phrases verbatim
        - Add subtle HALOS branding
        """
        
        # Cache to avoid API calls for repeat prompts
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt]
            
        response = await openai.Image.acreate(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        
        image_url = response['data'][0]['url']
        self.prompt_cache[prompt] = image_url
        return image_url

    async def split_command(self, chat_context: str) -> dict:
        """Parse /split commands from natural language"""
        completion = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Extract: total_amount, split_ratio, deadline from:"
            }, {
                "role": "user", 
                "content": chat_context
            }],
            temperature=0
        )
        
        return eval(completion.choices[0].message.content)  # Returns dict