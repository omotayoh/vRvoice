import grpc
from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc

RIVA_SERVER = "localhost:50051"

channel = grpc.insecure_channel(RIVA_SERVER)
asr_stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)

def _wait_for_channel_ready(timeout_sec=5):
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_sec)
        return True
    except grpc.FutureTimeoutError:
        return False

def _resolve_encoding_value():
    """
    Resolve the enum value for LINEAR_PCM across Riva variants.
    Prefers enums exposed via riva_asr_pb2, with fallbacks.
    """
    # Preferred location in your environment:
    try:
        audio_pb2 = riva_asr_pb2.riva_dot_proto_dot_riva__audio__pb2
        try:
            return audio_pb2.AudioEncoding.Value("LINEAR_PCM")
        except Exception:
            return audio_pb2.AudioEncoding.LINEAR_PCM
    except Exception:
        pass

    # Fallbacks for other builds:
    for candidate in (
        ("AudioEncoding", "Value"),
        ("AudioEncoding", None),
        ("RecognitionConfig.AudioEncoding", "Value"),
        ("RecognitionConfig.AudioEncoding", None),
    ):
        try:
            obj = riva_asr_pb2
            for part in candidate[0].split("."):
                obj = getattr(obj, part)
            if candidate[1] == "Value":
                return obj("LINEAR_PCM")
            return obj.LINEAR_PCM
        except Exception:
            continue

    # Last resort: typical numeric value for LINEAR_PCM
    return 1

def _config(sample_rate_hz: int, language_code: str):
    encoding_value = _resolve_encoding_value()
    return riva_asr_pb2.RecognitionConfig(
        encoding=encoding_value,               # your protos use 'encoding'
        sample_rate_hertz=sample_rate_hz,
        language_code=language_code,
        audio_channel_count=1,
        max_alternatives=1,
        enable_automatic_punctuation=True,
    )

def transcribe(audio_buffer: bytes, sample_rate_hz: int = 16000, language_code: str = "en-US") -> str:
    """
    Batch transcription: audio_buffer must be RAW 16-bit PCM (LE), mono, at sample_rate_hz.
    """
    if not _wait_for_channel_ready():
        print(f"Could not connect to Riva server at {RIVA_SERVER} within timeout.")
        return ""

    cfg = _config(sample_rate_hz, language_code)
    req = riva_asr_pb2.RecognizeRequest(config=cfg, audio=audio_buffer)

    try:
        response = asr_stub.Recognize(req)
        if not response.results:
            return ""
        return " ".join(
            result.alternatives[0].transcript
            for result in response.results
            if result.alternatives
        )
    except grpc.RpcError as e:
        print(f"Riva ASR error: code={e.code()}, details={e.details()}")
        return ""

def stream_transcribe(chunk_generator, sample_rate_hz: int = 16000, language_code: str = "en-US", append_silence_tail_sec: float = 0.5):
    """
    Streaming transcription. chunk_generator yields dicts: {"audio_content": bytes}.
    Sends streaming config first, then audio chunks. Optionally appends a silence tail.
    Yields dicts: {"text": str, "is_final": bool}.
    """
    if not _wait_for_channel_ready():
        print(f"Could not connect to Riva server at {RIVA_SERVER} within timeout.")
        return

    cfg = _config(sample_rate_hz, language_code)

    def request_stream():
        # First message: streaming config (NO single_utterance in your schema)
        yield riva_asr_pb2.StreamingRecognizeRequest(
            streaming_config=riva_asr_pb2.StreamingRecognitionConfig(
                config=cfg,
                interim_results=True,
            )
        )
        # Audio chunks
        for item in chunk_generator:
            audio_bytes = item.get("audio_content", b"")
            if audio_bytes:
                yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=audio_bytes)
        # Optional: silence tail to help endpointing finalize
        if append_silence_tail_sec and append_silence_tail_sec > 0:
            tail_samples = int(append_silence_tail_sec * sample_rate_hz)
            tail_silence = b"\x00\x00" * tail_samples  # 16-bit PCM LE zeros
            yield riva_asr_pb2.StreamingRecognizeRequest(audio_content=tail_silence)

    try:
        responses = asr_stub.StreamingRecognize(request_stream())
        for response in responses:
            for result in response.results:
                if result.alternatives:
                    yield {
                        "text": result.alternatives[0].transcript,
                        "is_final": result.is_final,
                    }
    except grpc.RpcError as e:
        print(f"Riva streaming error: code={e.code()}, details={e.details()}")