#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from src.utils.audio import resample_audio_to_16khz

load_dotenv()

test_file_path = Path(__file__).parent / "tests" / "data" / "1735404531.458927.mp3"

print("Loading diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.getenv("HF_TOKEN")
).to(torch.device("cpu"))

print(f"Original audio file: {test_file_path}")
print(f"Resampling to 16 kHz...")
resampled_path = resample_audio_to_16khz(str(test_file_path))
print(f"Resampled file: {resampled_path}")

print("Running speaker diarization...")
try:
    diarization = diarization_pipeline(resampled_path)
    print("Diarization completed successfully!")
    print(f"Number of speakers detected: {len(set(diarization.labels()))}")
except Exception as e:
    print(f"Error during diarization: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up temp file
    if 'resampled_path' in locals() and resampled_path != str(test_file_path):
        Path(resampled_path).unlink(missing_ok=True)
        print(f"Cleaned up: {resampled_path}")
