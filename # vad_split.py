# vad_split.py
from pathlib import Path
import wave, contextlib, collections
import webrtcvad
import numpy as np
import soundfile as sf

in_wav  = Path(r"C:\Users\Tala\Downloads\output_recordings\session_16k.wav")
out_dir = Path(r"C:\Users\Tala\Downloads\output_recordings\clean")
out_dir.mkdir(parents=True, exist_ok=True)

# ========= VAD SETTINGS =========
FRAME_MS = 20          # was 30 → finer decisions
AGGRESSIVENESS = 3     # was 2 → stricter speech detection
PADDING_MS = 120       # was 300 → shorter hangover so pauses split
MIN_DUR = 0.4          # ok
MAX_DUR = 30.0         # ok (doesn't split; just a cap)


# ========= INTERNALS =========
class Frame(collections.namedtuple("Frame", ["bytes","timestamp","duration"])):
    __slots__ = ()

def read_wave(path: Path):
    # expects 16kHz mono 16-bit PCM (your preprocess did this)
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        assert wf.getnchannels() == 1, "expected mono; run preprocess first"
        assert wf.getsampwidth() == 2, "expected 16-bit PCM"
        assert wf.getframerate() == 16000, "expected 16k sample rate"
        frames = wf.getnframes()
        pcm = wf.readframes(frames)
        return pcm, wf.getframerate()

def frame_generator(frame_ms, audio_bytes, sample_rate):
    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0) * 2)  # 2 bytes/sample
    offset = 0
    timestamp = 0.0
    duration = frame_ms / 1000.0
    while offset + bytes_per_frame <= len(audio_bytes):
        yield Frame(audio_bytes[offset:offset+bytes_per_frame], timestamp, duration)
        timestamp += duration
        offset += bytes_per_frame

def vad_collect(sample_rate, frame_ms, pad_ms, vad, frames):
    num_padding = int(pad_ms / frame_ms)
    ring = collections.deque(maxlen=num_padding)
    triggered = False
    voiced = []
    start_time = 0.0

    for f in frames:
        speech = vad.is_speech(f.bytes, sample_rate)
        if not triggered:
            ring.append((f, speech))
            if sum(1 for _, s in ring if s) > 0.9 * ring.maxlen:
                triggered = True
                start_time = ring[0][0].timestamp
                voiced.extend(fr for fr, _ in ring)
                ring.clear()
        else:
            voiced.append(f)
            ring.append((f, speech))
            if sum(1 for _, s in ring if not s) > 0.9 * ring.maxlen:
                end_time = f.timestamp + f.duration
                yield b"".join(fr.bytes for fr in voiced), start_time, end_time
                voiced = []
                ring.clear()
                triggered = False

    if voiced:
        end_time = voiced[-1].timestamp + voiced[-1].duration
        yield b"".join(fr.bytes for fr in voiced), start_time, end_time

# ========= RUN =========
pcm, sr = read_wave(in_wav)
vad = webrtcvad.Vad(AGGRESSIVENESS)
frames = list(frame_generator(FRAME_MS, pcm, sr))

count = 0
for audio_bytes, t0, t1 in vad_collect(sr, FRAME_MS, PADDING_MS, vad, frames):
    dur = t1 - t0
    if dur < MIN_DUR or dur > MAX_DUR:
        continue
    # convert back to float32 in [-1,1] for saving
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    out_wav = out_dir / f"session_seg{count:04d}.wav"
    sf.write(out_wav, audio, sr, subtype="PCM_16")
    count += 1

print("segments written:", count, "->", out_dir)
