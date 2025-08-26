from pathlib import Path
from pydub import AudioSegment
import random

# config
input_dir = Path(r"C:\Users\Tala\Downloads\recordings")
output_session = Path(r"C:\Users\Tala\Downloads\output_recordings\session.wav")
output_enroll = Path(r"C:\Users\Tala\Downloads\enroll")


speakers = ["jackson", "george", "lucas"]  # update if you have different
per_speaker = 6
silence_ms = 300  # pause between utterances

# create silence chunk
silence = AudioSegment.silent(duration=silence_ms)

# collect utterances
segments = []
for spk in speakers:
    files = sorted([f for f in input_dir.glob(f"*{spk}*.wav")])[:per_speaker]
    random.shuffle(files)
    # save one for enrollment
    enroll_file = files[0]
    enroll_audio = AudioSegment.from_wav(enroll_file)
    out_dir = output_enroll / spk
    out_dir.mkdir(parents=True, exist_ok=True)
    enroll_audio.export(out_dir / "enroll.wav", format="wav")
    # rest go into session
    for f in files[1:]:
        seg = AudioSegment.from_wav(f)
        segments.append(seg + silence)

# shuffle all speaker turns, like a classroom dialog
random.shuffle(segments)

# combine
session = sum(segments)
output_session.parent.mkdir(parents=True, exist_ok=True)
session.export(output_session, format="wav")

print("Saved session:", output_session)
print("Saved enrollment clips in:", output_enroll)
