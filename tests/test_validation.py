from Validation import AudioValidator


def test_format():

    validator = AudioValidator("sample_audio/sample.wav")

    assert validator.detect_format() == "WAV"

