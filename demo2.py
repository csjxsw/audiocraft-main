import torchaudio

from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio

## for test

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=8)

melody_waveform, sr = torchaudio.load("assets/bach.mp3")
melody_waveform = melody_waveform.unsqueeze(0).repeat(2, 1, 1)
output = model.generate_with_chroma(
    descriptions=[
        '80s pop track with bassy drums and synth',
        '90s rock song with loud guitars and heavy drums',
    ],
    melody_wavs=melody_waveform,
    melody_sample_rate=sr,
    progress=True
)
display_audio(output, sample_rate=32000)