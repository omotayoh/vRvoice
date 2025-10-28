import sounddevice as sd
import queue
import threading
import time
import sys
import signal

from modules.asr_engine import stream_transcribe
from modules.nlu_engine import get_command
# from modules.vred_client import send_activate_vset  # Uncomment when ready

# --- CONFIG ---
ENABLE_VRED = True  # Change to True for full E2E test
VRED_HOST = "127.0.0.1"  # IP of the VRED machine (or "127.0.0.1" if same)
VRED_PORT = 8888
VRED_TOKEN = None

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = 'int16'
CHUNK_DURATION_SEC = 0.1  # 100 ms ≈ 3200 bytes at 16kHz mono 16-bit

# Queue holds raw PCM byte chunks from mic -> ASR streaming
audio_queue = queue.Queue(maxsize=50)
shutdown_event = threading.Event()


def audio_callback(indata, frames, time_info, status):
    """Realtime callback from sounddevice; push raw PCM bytes to queue."""
    if status:
        print(f"[audio] {status}", file=sys.stderr)
    try:
        audio_queue.put_nowait(bytes(indata))
    except queue.Full:
        print("[audio] Warning: audio queue full; dropping chunk", file=sys.stderr)


def chunk_generator():
    """Yield dicts of {'audio_content': bytes} for ASR streaming."""
    while not shutdown_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.5)
            if chunk:
                yield {"audio_content": chunk}
        except queue.Empty:
            continue


def process_audio_stream():
    """Consume final transcripts, route to NLU, and optionally activate VRED vset."""
    print("Starting real-time streaming transcription...")
    try:
        for transcript in stream_transcribe(
            chunk_generator(),
            sample_rate_hz=SAMPLE_RATE,
            language_code="en-US",
            append_silence_tail_sec=0.0,  # live mic: no synthetic tail
        ):
            if not transcript.get("is_final", False):
                continue

            text = transcript.get("text", "").strip()
            if not text:
                continue

            print(f"Final: {text}")

            # NLU routing
            vset, matched_phrase, score = get_command(text)
            if vset:
                group, name = vset[0], vset[1]
                print(f"[client] Matched: '{matched_phrase}' (score={score:.2f}) → VSet: {group}/{name}")

                if ENABLE_VRED:
                    try:
                        from modules.vred_client import send_activate_vset
                        resp = send_activate_vset(VRED_HOST, VRED_PORT, group, name, token=VRED_TOKEN, timeout=1.0)
                        if resp.get("ok"):
                            print(f"[client] Activated VSet: {group}/{name}")
                        else:
                            print(f"[client] VRED error: {resp}")
                    except ImportError:
                        print("[client] VRED client not available. Skipping call.")
                else:
                    print("[client] Skipping VRED call (demo mode)")
            else:
                print("No matching command found.")
    except KeyboardInterrupt:
        pass
    finally:
        print("[stream] Exiting streaming loop")


def main():
    print(f"Listening for voice commands... (Ctrl+C to stop)")
    print(f"ENABLE_VRED={ENABLE_VRED} | VRED target: {VRED_HOST}:{VRED_PORT}")

    def handle_sigint(sig, frame):
        print("\n[main] SIGINT received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    worker = threading.Thread(target=process_audio_stream, daemon=True)
    worker.start()

    blocksize = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=blocksize,
            dtype=DTYPE,
            channels=CHANNELS,
            callback=audio_callback,
        ):
            while not shutdown_event.is_set():
                time.sleep(0.2)
    except Exception as ex:
        print(f"[main] Audio device error: {ex}", file=sys.stderr)
    finally:
        shutdown_event.set()
        worker.join(timeout=2.0)
        print("[main] Shutdown complete.")


if __name__ == "__main__":
    main()