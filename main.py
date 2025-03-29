import speech_recognition as sr
from IPython.display import Audio as AudioPlayer
from dotenv import load_dotenv
import base64
import wave
from pydub import AudioSegment
from gtts import gTTS
load_dotenv()

# Audio Configuration
SAMPLE_RATE = 24000  # Hz (24kHz)
CHANNELS = 1  # Mono
SAMPLE_WIDTH = 2  # Bytes (16 bits)


from agno.agent import Agent
from agno.media import Audio
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
from agno.tools.desi_vocal import DesiVocalTools

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    markdown=True,
    add_history_to_messages=True,
    instructions=["You will receive an audio containing your prompt. Listening to it and answer",
                 "The person in speaking in spanish",
                 "After hearing the audio responde using text_to_speech tool"]
)