# accent-change-paper-implementation

Loose implementation of paper called "End-to-End Accent Conversion Without Using Native Utterances" using [OpenAI's Whisper model](https://huggingface.co/openai/whisper-small), [Coqui AI's voice synthesizer](https://github.com/coqui-ai/TTS), and [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) voice embedder. 

Link to original paper: https://ieeexplore.ieee.org/document/9053797

## setup
Source the `saved_models` directory from this implementation: https://github.com/CorentinJ/Real-Time-Voice-Cloning?tab=readme-ov-file

## usage
```
usage: accent_changer.py [-h] [-i INPUT] [-o OUTPUT] [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
  -o OUTPUT, --output OUTPUT
  --input_dir INPUT_DIR
  --output_dir OUTPUT_DIR
```
