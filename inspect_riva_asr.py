from riva.client.proto import riva_asr_pb2

print("Top-level names on riva_asr_pb2:")
print([n for n in dir(riva_asr_pb2) if not n.startswith("_")])

print("\nRecognitionConfig descriptor:")
print(riva_asr_pb2.RecognitionConfig)

# Field names present on RecognitionConfig:
print("\nRecognitionConfig fields:")
print(list(riva_asr_pb2.RecognitionConfig.DESCRIPTOR.fields_by_name.keys()))

# See if an enum is nested:
print("\nNested types on RecognitionConfig:")
print([n.name for n in riva_asr_pb2.RecognitionConfig.DESCRIPTOR.nested_types])

# If nested enum exists, list its values
if hasattr(riva_asr_pb2.RecognitionConfig, "AudioEncoding"):
    enc = riva_asr_pb2.RecognitionConfig.AudioEncoding
    print("\nNested AudioEncoding values (if present):")
    try:
        print(enc.values_by_name.keys())
    except Exception:
        pass