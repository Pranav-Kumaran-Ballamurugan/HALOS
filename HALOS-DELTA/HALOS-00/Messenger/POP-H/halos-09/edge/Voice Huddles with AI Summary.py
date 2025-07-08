# halos/voice/huddle.py
import whisper
from pyannote.audio import Pipeline

class VoiceHuddle:
    def __init__(self):
        self.transcriber = whisper.load_model("small")
        self.diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization")
        
    async def start_huddle(self, participant_ids: List[str]):
        """Begin encrypted voice session"""
        self.active_speakers = {id: [] for id in participant_ids}
        self.raw_audio = []
        
        # WebRTC setup would go here
        print(f"Huddle started with {participant_ids}")

    async def transcribe_segment(self, audio_segment: bytes):
        """Process real-time audio"""
        # Speaker diarization
        diarization = self.diarizer(audio_segment)
        
        # Whisper transcription
        result = self.transcriber.transcribe(audio_segment)
        
        # Map speakers to HALOS IDs
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_id = self._match_voiceprint(speaker)
            text = result["text"][turn.start:turn.end]
            self.active_speakers[speaker_id].append(text)

    async def end_huddle(self) -> str:
        """Generate smart summary"""
        full_transcript = chr(10).join(
            f"{speaker}: {text}"
            for speaker, texts in self.active_speakers.items()
            for text in texts
        )
        
        # Extract key decisions
        summary = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Extract concrete plans, deadlines, and amounts from:"
            }, {
                "role": "user",
                "content": full_transcript
            }]
        )
        
        return summary.choices[0].message.content