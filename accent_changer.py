import argparse
import soundfile as sf

from voice_embedder import embed_voices
import ASR
from MS_TTS import synth_audio

if __name__ == "__main__":
    # TODO: add -h descriptions
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output", required=False, default="out.wav")
    
    args = parser.parse_args()
    
    # get voice embedding
    voice_embedding = embed_voices([args.input])[0]
    
    # print(len(voice_embedding[0])) # DEBUG
    
    # get content of recording
    content = ASR.annotate_files([args.input])[0]['text']
    
    # print("CONTENT:", content) # DEBUG
    
    # generate output
    audio_array, sample_rate = synth_audio(content, voice_embedding)
    
    # save output
    out_path = args.output
    sf.write(out_path, audio_array, sample_rate)
    print()
    print(f"Output saved to {out_path}")
    