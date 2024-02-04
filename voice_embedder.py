from resemblyzer import VoiceEncoder, preprocess_wav


encoder = VoiceEncoder()
    

def embed_voices(voices: list(str)) -> list:
    # load and preprocess audio tracks
    for i in range(len(voices)):
        voices[i] = preprocess_wav(voices[i])

    # embed utterances (semantics of audio)
    for i in range(len(voices)):
        voices[i] = encoder.embed_speaker([voices[i]])
        
    return voices
