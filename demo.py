import pickle
import sys

import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import torch
from PTalker import PTalker
from transformers import AutoProcessor
import librosa

from sklearn.manifold import TSNE

import matplotlib.backends
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def remove_parametrizations(state_dict):
    keys_to_remove = []
    for key in state_dict.keys():
        if "parametrizations" in key:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict

@torch.no_grad()
def test(args, model, wav, path, epoch):
    dev = args.device

    result_path = os.path.join(args.dataset, args.result_path)

    wav_path = "F1_e23.wav"
    processor = AutoProcessor.from_pretrained("hubert-large-ls960-ft")
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    audio = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)

    test_subjects_list = [i for i in args.train_subjects.split(" ")]


    model.load_state_dict(torch.load("", map_location='cpu'))


    model = model.to(args.device)
    model.eval()

    one_hot_all = np.eye(len(args.train_subjects.split(' ')))

    template_file = "/BIWI/templates.pkl"
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    for subject in templates.keys():
        print(subject)
    # "F1 F5 F6 F7 F8 M1 M2 M6"
    subject = 'F1'
    template = templates[subject]
    one_hot = one_hot_all[-1, :]
    condition_subject = test_subjects_list[-1]


    ref_motion = np.load("")  # (frame_num, V*3)
    ref_motion = torch.FloatTensor(ref_motion)  # to tensor
    ref_motion = ref_motion.unsqueeze(0).to(dev)

    audio = torch.FloatTensor(audio).unsqueeze(0).to(dev)
    template = torch.FloatTensor(template).unsqueeze(0).flatten(-2).to(dev)
    one_hot = torch.FloatTensor(one_hot).unsqueeze(0).to(dev)

    prediction = model.predict_seen(audio, template, one_hot)

    prediction = prediction.squeeze()  # (seq_len, V*3)
    np.save(os.path.join(result_path, wav[:-4] + "_subject_" + subject + "_condition_" + condition_subject + ".npy"),
            prediction.detach().cpu().numpy())

def main():
    parser = argparse.ArgumentParser(
        description='PTalker: Personalized Speech-Driven 3D Talking Head Animation via Style Disentanglement and Modality Alignment')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370*3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="BIWI/audio/wav_e",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="BIWI/vertices_npy",
                        help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=100, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--template_file", type=str, default="BIWI/templates.pkl",
                        help='path of the personalized templates')
    parser.add_argument("--result_path", type=str,
                        default="BIWI/result",
                        help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")  #BIWI-Test-B
    parser.add_argument("--test_wav_path", type=str, default='demo/wav')
    parser.add_argument("--lip_region", type=str, default="/BIWI/regions/lip.txt",
                        help='path to the lip region')
    args = parser.parse_args()

    # build model
    model = PTalker(args)

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    demo_audio = args.test_wav_path

    for file in os.listdir(demo_audio):
        test(args, model, file, args.test_wav_path, epoch=100)


if __name__ == '__main__':
    main()