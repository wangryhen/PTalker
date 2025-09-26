import sys

import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import torch
import pickle
import torch.nn as nn
from data_loader import get_dataloaders

from PTalker import PTalker

os.environ["CUDA_LAUNCH_BLOCKING"] = '5'


def remove_parametrizations(state_dict):
    keys_to_remove = []
    for key in state_dict.keys():
        if "parametrizations" in key:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del state_dict[key]
    return state_dict


@torch.no_grad()
def test(args, model, test_loader, epoch):
    dev = args.device

    result_path = "/BIWI/official_npy"
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    # save_path = os.path.join(args.dataset, args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    state_dict = torch.load(
        "official_pretrained_model/BIWI_model.pth", map_location='cpu')
    state_dict = remove_parametrizations(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    model.eval()

    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # to gpu
        audio, vertice, template, one_hot_all = audio.to(dev), vertice.to(dev), template.to(dev), one_hot_all.to(dev)
        test_subject = "_".join(file_name[0].split("_")[:-1])
        if test_subject in train_subjects_list:
            condition_subject = test_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()  # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze()  # (seq_len, V*3)
                np.save(
                    os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"),
                    prediction.detach().cpu().numpy())

def metric(args):
    train_subject_list = args.train_subjects.split(" ")
    sentence_list = ["e" + str(i).zfill(2) for i in range(37, 41)]

    with open(args.templates_path, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    with open(os.path.join(args.region_path, "lve.txt")) as f:
        maps = f.read().split(", ")
        mouth_map = [int(i) for i in maps]

    with open(os.path.join(args.region_path, "fdd.txt")) as f:
        maps = f.read().split(", ")
        upper_map = [int(i) for i in maps]

    cnt = 0
    vertices_gt_all = []
    vertices_pred_all = []
    motion_std_difference = []

    for subject in train_subject_list:
        for sentence in sentence_list:
            vertices_gt = np.load(os.path.join(args.gt_path, subject + "_" + sentence + ".npy")).reshape(-1, 23370, 3)
            vertices_pred = np.load(
                os.path.join(args.pred_path, subject + "_" + sentence + "_condition_" + subject + ".npy")) \
                .reshape(-1, 23370, 3)
            vertices_pred = vertices_pred[:vertices_gt.shape[0], :, :]
            vertices_gt = vertices_gt[:vertices_pred.shape[0], :, :]

            motion_pred = vertices_pred - templates[subject].reshape(1, 23370, 3)
            motion_gt = vertices_gt - templates[subject].reshape(1, 23370, 3)

            cnt += vertices_gt.shape[0]

            vertices_gt_all.extend(list(vertices_gt))
            vertices_pred_all.extend(list(vertices_pred))

            L2_dis_upper = np.array([np.square(motion_gt[:, v, :]) for v in upper_map])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
            L2_dis_upper = np.sum(L2_dis_upper, axis=2)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            gt_motion_std = np.mean(L2_dis_upper)

            L2_dis_upper = np.array([np.square(motion_pred[:, v, :]) for v in upper_map])
            L2_dis_upper = np.transpose(L2_dis_upper, (1, 0, 2))
            L2_dis_upper = np.sum(L2_dis_upper, axis=2)
            L2_dis_upper = np.std(L2_dis_upper, axis=0)
            pred_motion_std = np.mean(L2_dis_upper)

            motion_std_difference.append(gt_motion_std - pred_motion_std)

    print('Frame Number: {}'.format(cnt))

    vertices_gt_all = np.array(vertices_gt_all)
    vertices_pred_all = np.array(vertices_pred_all)

    L2_dis_mouth_max = np.array([np.square(vertices_gt_all[:, v, :] - vertices_pred_all[:, v, :]) for v in mouth_map])
    L2_dis_mouth_max = np.transpose(L2_dis_mouth_max, (1, 0, 2))
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)

    print('Lip Vertex Error: {:.4e}'.format(np.mean(L2_dis_mouth_max)))
    print('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))


def main():
    parser = argparse.ArgumentParser(
        description='PTalker: Personalized Speech-Driven 3D Talking Head Animation via Style Disentanglement and Modality Alignment')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370 * 3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="BIWI/audios/wav_e",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str,
                        default="/BIWI/vertices_npy",
                        help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:7")

    parser.add_argument("--template_file", type=str, default="/BIWI/templates.pkl",
                        help='path of the personalized templates')
    parser.add_argument("--result_path", type=str,
                        default="BIWI/official_npy",
                        help='path to the predictions')
    parser.add_argument("--lip_region", type=str, default="BIWI/regions/lve.txt",
                        help='path to the lip region')
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str,
                        default="F2 F3 F4 M3 M4 M5")  # BIWI-Test-A, 24 sequences (6 subjects 4 sentences)
    parser.add_argument("--pred_path", type=str,
                        default="BIWI/result/")
    parser.add_argument("--gt_path", type=str, default="BIWI/vertices_npy")
    parser.add_argument("--region_path", type=str, default="BIWI/regions/")
    parser.add_argument("--templates_path", type=str, default="BIWI/templates.pkl")

    args = parser.parse_args()

    # build model
    model = PTalker(args)

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    # load data
    dataset = get_dataloaders(args)

    test(args, model, dataset["test"], epoch='best')

    metric(args)


if __name__ == '__main__':
    main()