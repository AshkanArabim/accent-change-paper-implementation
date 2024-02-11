# NOTE: most code here was gathered from 
# https://github.com/CorentinJ/Real-Time-Voice-Cloning

import librosa
import numpy as np
import soundfile as sf
import torch
from pathlib import Path


from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder


PARAMS = {
    # 'enc_model_fpath': Path('./saved_models/default/encoder.pt'),
    'voc_model_fpath': Path('./saved_models/default/vocoder.pt'),
    'syn_model_fpath': Path('./saved_models/default/synthesizer.pt'),
}


# input: text from whisper, audio file for voice embedding
# output: synthesized audio file
def synth_audio(s: str, voice_embedding):
    vocoder.load_model(PARAMS['voc_model_fpath'])
    
    # note: not sure if I should L2 normalize the voice_embedding? we'll see...
    # note: synthesizer.synthesize_spectrograms() can receive multiple texts and
    # embeds if you want to do batch processing.
    synthesizer = Synthesizer(PARAMS['syn_model_fpath'])
    texts = [s]
    embeds = [voice_embedding]
    # print("EMBEDS:", type(embeds[0])) # DEBUG
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    
    # generate audio
    audio = vocoder.infer_waveform(specs[0])
    # pad output (supposedly there's an early-cutoff bug?)
    audio = np.pad(audio, (0, synthesizer.sample_rate), mode="constant")
    
    # note: I am not trimming excess silence like the original code. I'll do it if necessary
    
    audio_array = audio.astype(np.float32)
    sample_rate = synthesizer.sample_rate
    return audio_array, sample_rate
    
    # # TODO: remove this last part; should be done outside
    # # save generated output
    # sf.write(out_path, audio.astype(np.float32), synthesizer.sample_rate)
    # print(f"saved to {out_path}")
