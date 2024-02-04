import torch
import datasets
from transformers import pipeline


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIPE = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=DEVICE)


def annotate_samples(dataset, subset='train'):
    samples = [dataset[subset][i]['audio'] for i in range(len(dataset['train']))]
    return PIPE(samples) # There is no limit on the number of tokens
    
    
def annotate_folder(path: str):
    dataset = datasets.load_dataset("audiofolder", data_dir=path)
    return annotate_samples(dataset)


def annotate_files(paths: list[str]):
    # manually adding "train" to match result produced by datasets.load_dataset
    dataset = {"train": datasets.Dataset.from_dict({"audio": paths}).cast_column("audio", datasets.Audio())}
    return annotate_samples(dataset)


# example use: 
# print(annotate_folder('./data'))
# print(annotate_files(['./data/ali accent.wav', './data/ashkan bad accent.wav']))
