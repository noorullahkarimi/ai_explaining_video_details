from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import torch
#3th page
processor_speech = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model_speech = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

speaker_index = 0
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)

def speak(text_input):
    inputs = processor_speech(text=text_input, return_tensors="pt")
    speech = model_speech.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
    return speech.cpu().numpy()