import os
import wave
import librosa
import numpy as np
import soundfile as sf
from mutagen import File


class AudioValidator:

    def __init__(self, file_path):
        self.file_path = file_path

    def validate_header(self):
        signatures = {
            b'RIFF': 'WAV',
            b'ID3': 'MP3',
            b'fLaC': 'FLAC'
        }

        with open(self.file_path, 'rb') as file:
            header = file.read(4)

        for sig, fmt in signatures.items():
            if header.startswith(sig):
                return fmt

        if header[0] == 0xFF:
            return 'AAC'

        return 'Unknown'

    def detect_format(self):
        ext = os.path.splitext(self.file_path)[1].lower()
        return ext.replace(".", "").upper()

    def get_metadata(self):
        audio = File(self.file_path)

        if audio:
            return {
                "duration": audio.info.length,
                "sample_rate": audio.info.sample_rate
            }

        return {}

    def wav_info(self):
        if self.file_path.endswith(".wav"):
            with wave.open(self.file_path, 'rb') as wav:
                return {
                    "channels": wav.getnchannels(),
                    "sample_width": wav.getsampwidth(),
                    "frame_rate": wav.getframerate()
                }

        return {}

    def load_audio(self):
        y, sr = librosa.load(self.file_path, sr=None)
        return y, sr

    def detect_bass_energy(self):
        y, sr = self.load_audio()

        fft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        bass_band = fft[(freqs >= 20) & (freqs <= 250)]
        return bass_band.mean()

    def detect_treble_energy(self):
        y, sr = self.load_audio()

        fft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        treble_band = fft[(freqs >= 4000) & (freqs <= 20000)]
        return treble_band.mean()

    def detect_clipping(self):
        y, sr = self.load_audio()

        clipped_samples = np.sum(np.abs(y) >= 0.99)
        return clipped_samples

    def detect_reverb(self):
        y, sr = self.load_audio()

        energy = y ** 2
        return energy.mean()

    def full_report(self):
        report = {
            "header_format": self.validate_header(),
            "extension_format": self.detect_format(),
            "metadata": self.get_metadata(),
            "wav_info": self.wav_info(),
            "bass_energy": self.detect_bass_energy(),
            "treble_energy": self.detect_treble_energy(),
            "clipping_count": self.detect_clipping(),
            "reverb_estimation": self.detect_reverb()
        }

        return report
    

Validate = AudioValidator("/Users/shaikmohammadbude/SNR/traffic_speech.wav")
report = Validate.full_report()
print("\nAUDIO VALIDATION REPORT")
print("-" * 40)

for key, value in report.items():
    print(f"{key}: {value}")

Validate2 = AudioValidator("/Users/shaikmohammadbude/SNR/file_example_MP3_5MG.mp3")
report = Validate2.full_report()
print("\nAUDIO VALIDATION REPORT")
print("-" * 40)

for key, value in report.items():
    print(f"{key}: {value}")