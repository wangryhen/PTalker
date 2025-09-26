import sys
from fvcore.nn import FlopCountAnalysis
import time
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import torch
import torch.nn as nn
import random
from data_loader import get_dataloaders
from PTalker import PTalker

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import os
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from termcolor import colored
# import torch.optim.lr_scheduler as lr_scheduler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


def seed_torch(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(seed=7)



writer = {
    'Train_e_loss': SummaryWriter("BIWI/logs/Train_e_loss"),
    'Train_i_loss': SummaryWriter("BIWI/logs/Train_i_loss"),
    'Valid_e_loss': SummaryWriter("BIWI/logs/Valid_e_loss")
}


def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    dev = args.device
    save_path = os.path.join(args.dataset, args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    initial_sampling_rate = 1.0
    decay_factor = 0.05

    min_loss_teacher = 1000
    min_loss_non_teacher = 1000
    min_train_loss = 1000
    best_train_epoch = -1
    best_val_epoch_teacher = -1
    best_val_epoch_non_teacher = -1

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0

    scaler = GradScaler()

    for e in range(epoch + 1):
        sampling_rate = initial_sampling_rate * (np.exp(-decay_factor * e))
        print(f"Epoch {e + 1}, Sampling Rate: {sampling_rate}")

        loss_log = []
        # train
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()

        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1
            # to gpu
            audio, vertice, template, one_hot = audio.to(dev), vertice.to(dev), template.to(dev), one_hot.to(dev)

            with autocast():
                loss_rec, loss_constraints, loss_detail, v_pre, v_gt = model(audio, template, vertice,
                                                                                       one_hot, criterion,
                                                                                       teacher_forcing=True,
                                                                                       sampling_rate=sampling_rate)
                prediction_shift = v_pre[:, 1:, :] - v_pre[:, :-1, :]
                target_shift = v_gt[:, 1:, :] - v_gt[:, :-1, :]
                vel_loss = torch.mean((criterion(prediction_shift, target_shift)))
                loss = loss_rec + 1.2 * loss_detail + 10 * vel_loss + 0.01 * loss_constraints
            scaler.scale(loss).backward()
            # loss.backward()
            loss_log.append(loss.item())
            if i % args.gradient_accumulation_steps == 0:
                # optimizer.step()
                # optimizer.zero_grad()
                scaler.step(optimizer)  # Update weights
                scaler.update()  # Update the scale for next iteration
                optimizer.zero_grad()

            writer['Train_i_loss'].add_scalar("train_iteration_loss", loss.item(), iteration)

            pbar.set_description(
                "(Epoch {} / {}, iteration {}) TRAIN LOSS:{:.8f}".format((e + 1), epoch, iteration, np.mean(loss_log)))

        avg_train_loss = np.mean(loss_log)
        writer['Train_e_loss'].add_scalar("train_epoch_loss", avg_train_loss, e)

        # Update best training loss and epoch
        if avg_train_loss < min_train_loss:
            min_train_loss = avg_train_loss
            best_train_epoch = e

        # validation
        with torch.no_grad():
            valid_loss_log = []
            model.eval()
            for audio, vertice, template, one_hot_all, file_name in dev_loader:
                # to gpu
                audio, vertice, template, one_hot_all = audio.to(dev), vertice.to(dev), template.to(
                    dev), one_hot_all.to(dev)
                train_subject = "_".join(file_name[0].split("_")[:-1])
                if train_subject in train_subjects_list:
                    condition_subject = train_subject
                    iter = train_subjects_list.index(condition_subject)
                    one_hot = one_hot_all[:, iter, :]

                    # Forward pass with autocast for validation
                    with autocast():
                        loss_rec, loss_constraints, loss_detail, v_pre, v_gt = model(audio, template, vertice,
                                                                                               one_hot, criterion,
                                                                                               teacher_forcing=True,
                                                                                               sampling_rate=sampling_rate)
                        prediction_shift = v_pre[:, 1:, :] - v_pre[:, :-1, :]
                        target_shift = v_gt[:, 1:, :] - v_gt[:, :-1, :]
                        vel_loss = torch.mean((criterion(prediction_shift, target_shift)))
                        loss = loss_rec + 1.2 * loss_detail + 10 * vel_loss + 0.01 * loss_constraints
                        valid_loss_log.append(loss.item())
                else:
                    for iter in range(one_hot_all.shape[-1]):
                        condition_subject = train_subjects_list[iter]
                        one_hot = one_hot_all[:, iter, :]

                        # Forward pass with autocast for validation
                        with autocast():
                            loss_rec, loss_constraints, loss_detail, v_pre, v_gt = model(audio, template,
                                                                                                   vertice, one_hot,
                                                                                                   criterion,
                                                                                                   teacher_forcing=True,
                                                                                                   sampling_rate=sampling_rate)
                            prediction_shift = v_pre[:, 1:, :] - v_pre[:, :-1, :]
                            target_shift = v_gt[:, 1:, :] - v_gt[:, :-1, :]
                            vel_loss = torch.mean((criterion(prediction_shift, target_shift)))
                            loss = loss_rec + loss_detail + 10 * vel_loss + 0.01 * loss_constraints
                            valid_loss_log.append(loss.item())

        current_loss = np.mean(valid_loss_log)


        current_lr = optimizer.param_groups[0]['lr']
        print(colored(f"Epoch {e + 1}: Learning Rate = {current_lr}", "yellow"))

        writer['Valid_e_loss'].add_scalar("val_epoch_loss", current_loss, e)

        if save_path is None:
            os.makedirs(save_path)


        if sampling_rate > 0:
            if current_loss < min_loss_teacher:
                cs_loss = min_loss_teacher - current_loss
                print(colored("saving best teacher forcing model...", "green"))
                print(colored(f"current loss: {current_loss}", "red"))
                min_loss_teacher = current_loss
                best_val_epoch_teacher = e
                torch.save(model.state_dict(), os.path.join(save_path, 'best_teacher_model.pth'))
                if (e > 15 and cs_loss > 0.000000000001):
                    torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{e}_{cs_loss}_teacher_model.pth'))

        else:
            if current_loss < min_loss_non_teacher:
                cs_loss = min_loss_non_teacher - current_loss
                print(colored("saving best non-teacher forcing model...", "green"))
                print(colored(f"current loss: {current_loss}", "red"))
                min_loss_non_teacher = current_loss
                best_val_epoch_non_teacher = e
                torch.save(model.state_dict(), os.path.join(save_path, 'best_non_teacher_model.pth'))
                if cs_loss > 0.000000001:
                    torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{e}_{cs_loss}_non_teacher_model.pth'))

        if (e > 230 and e % 10 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))

        print(colored("epoch: {}, current loss: {:.7f}".format(e + 1, current_loss), "blue"))

    # Print the best train and validation loss and corresponding epochs
    print("Training completed.")
    print(f"Best teacher forcing validation loss: {min_loss_teacher:.7f} at epoch {best_val_epoch_teacher + 1}")
    print(f"Best non-teacher forcing validation loss: {min_loss_non_teacher:.7f} at epoch {best_val_epoch_non_teacher + 1}")

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %80s %9s %20s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %80s %9s %20g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def main():
    parser = argparse.ArgumentParser(
        description='PTalker: Personalized Speech-Driven 3D Talking Head Animation via Style Disentanglement and Modality Alignment')
    parser.add_argument("--lr", type=float, default=0.0003, help='learning rate')
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370 * 3,
                        help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default="BIWI/audios/wav_e",
                        help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="BIWI/vertices_npy",
                        help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=800, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--template_file", type=str, default="BIWI/templates.pkl",
                        help='path of the personalized templates')
    parser.add_argument("--save_path", type=str,
                        default="",
                        help='path of the trained models')
    parser.add_argument("--result_path", type=str,
                        default="",
                        help='path to the predictions')
    parser.add_argument("--lip_region", type=str, default="BIWI/regions/lve.txt",
                        help='path to the lip region')
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    # parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--test_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    args = parser.parse_args()

    # build model
    model = PTalker(args)
    print("model parameters: ", count_parameters(model))
    model_info(model)

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(args.device)

    # load data
    dataset = get_dataloaders(args)
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model = trainer(args, dataset["train"], dataset["test"], model, optimizer, criterion, epoch=args.max_epoch)
    # print(model)

if __name__ == "__main__":
    main()