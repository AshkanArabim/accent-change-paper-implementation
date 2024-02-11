import argparse
import torch
from TTS.api import TTS
import os

import ASR

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    # TODO: add -h descriptions
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output", required=False, default="out.wav")
    parser.add_argument("--input_dir", required=False)
    parser.add_argument("--output_dir", required=False, default="out")
    
    args = parser.parse_args()
    
    if args.input_dir:
        # we'll assume output_dir is provided
        os.makedirs(args.output_dir, exist_ok=True)
        
        source_names = [source_name for source_name in os.listdir(args.input_dir) if ".wav" in source_name]
        source_paths = [os.path.join(args.input_dir, source_name) for source_name in source_names]
        out_paths = [os.path.join(args.output_dir, source_name) for source_name in source_names]
        # get content of all recordings
        contents = [content['text'] for content in ASR.annotate_files(source_paths)]
        for i in range(len(source_names)):
            tts.tts_to_file(text=contents[i], speaker_wav=source_paths[i], language="en", file_path=out_paths[i])
    else:
        # get content of recording
        content = ASR.annotate_files([args.input])[0]['text']
        
        # generate and save output
        tts.tts_to_file(text=content, speaker_wav=args.input, language="en", file_path=args.output)
