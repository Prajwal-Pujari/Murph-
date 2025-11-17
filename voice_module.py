# # voice_module.py
# import sounddevice as sd
# import numpy as np
# import whisper
# import pyttsx3
# import torch

# # Initialize text-to-speech
# tts = pyttsx3.init()
# tts.setProperty("rate", 175)
# voices = tts.getProperty("voices")
# tts.setProperty("voice", voices[0].id)

# # Load Whisper model (use "small" or "base" for speed)
# print("🔹 Loading Whisper model... (this might take a few seconds)")
# # Using .en models is faster and more accurate if you only need English
# model = whisper.load_model("small.en")

# def record_audio_data(duration=5, samplerate=16000):
#     """
#     Records audio for a given duration and returns it as a NumPy array.
#     """
#     print("🎤 Listening... Speak now!")
#     # Record audio data as a float32 NumPy array
#     audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
#     sd.wait()
#     # The audio data is already in the correct format, just flatten it
#     return audio_data.flatten()

# def transcribe_audio_data(audio_data):
#     """
#     Transcribes audio data directly from a NumPy array using the Whisper model.
#     """
#     print("🧠 Transcribing...")
#     # Pass the NumPy array directly to the transcribe function
#     result = model.transcribe(audio_data, fp16=torch.cuda.is_available())
#     text = result["text"].strip()
#     print(f"🗣 You said: {text}")
#     return text

# def speak(text):
#     """
#     Speaks a text string using pyttsx3.
#     """
#     print(f"🤖 AI says: {text}")
#     tts.say(text)
#     tts.runAndWait()

# if __name__ == "__main__":
#     while True:
#         # 1. Record audio directly into a variable (NumPy array)
#         audio_data = record_audio_data(duration=6)

#         # 2. Pass the audio data to the transcription function
#         user_text = transcribe_audio_data(audio_data)

#         # Ensure user_text is not empty before processing
#         if user_text:
#             if any(word in user_text.lower() for word in ["exit", "quit", "stop"]):
#                 speak("Goodbye!")
#                 break

#             # Temporary: echo the user's speech back
#             response = f"You said: {user_text}"
#             speak(response)