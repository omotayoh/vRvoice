import sounddevice as sd
import wave
import time
from modules.asr_engine import transcribe, stream_transcribe

# Settings
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5  # seconds
OUTPUT_FILE = "sample.wav"

# --- Step 1: Record Audio ---
print(f"Recording {DURATION} seconds of audio...")
audio_data = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype='int16'
)
sd.wait()
print("Recording complete. Saving file...")

with wave.open(OUTPUT_FILE, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_data.tobytes())

print(f"Audio saved as {OUTPUT_FILE}\n")

# --- Step 2: Single-Shot Transcription ---
print("=== Testing Single-Shot Transcription ===")
with wave.open(OUTPUT_FILE, "rb") as wf:
    assert wf.getframerate() == SAMPLE_RATE
    assert wf.getnchannels() == CHANNELS
    audio_bytes = wf.readframes(wf.getnframes())  # RAW PCM frames
text = transcribe(audio_bytes, sample_rate_hz=SAMPLE_RATE, language_code="en-US")
print(f"Single-Shot Result: {text}\n")

# --- Step 3: Streaming Transcription ---
print("=== Testing Streaming Transcription ===")

def chunk_generator():
    try:
        with wave.open(OUTPUT_FILE, "rb") as wf:
            fr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            if fr != SAMPLE_RATE or ch != CHANNELS or sw != 2:
                print(f"[gen] WAV format mismatch: fr={fr}, ch={ch}, sw={sw}")
                return
            frames_per_chunk = int(0.1 * SAMPLE_RATE)  # ~100ms (~3200 bytes)
            while True:
                data = wf.readframes(frames_per_chunk)
                if not data:
                    break
                yield {"audio_content": data}
                time.sleep(0.1)  # simulate real-time pacing
    except Exception as ex:
        print(f"[gen] Exception opening/reading WAV: {ex}")
        return

for transcript in stream_transcribe(chunk_generator(), sample_rate_hz=SAMPLE_RATE, language_code="en-US"):
    if transcript["is_final"]:
        print(f"Final: {transcript['text']}")