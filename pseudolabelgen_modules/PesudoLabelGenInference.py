import torch
import argparse
import json

import numpy as np
import random
from pseudolabelgen_dataloaders import PseudoLabelDataLoader
from pseudolabelgen_trainer import Trainer
from pseudolabelgen_models import PseudoLabelGen
from torch import sigmoid

import pickle


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='../data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='../annotation_creation/annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='../files/iu_xray/labels_14.pickle',
                        help='the path to the directory containing the data.')

    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    
    args = parser.parse_args()
    return args


def main():

    args = parse_agrs()

    # build pseudolabel generator model architecture
    model = PseudoLabelGen(args)

    #set path
    IMG_CLASSIFIER_PATH = "pseudolabelgen_results_temp/iu_xray/model_best.pth"

    # Load
    #pdb.set_trace()

    def _prepare_device(n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    device, device_ids = _prepare_device(args.n_gpu)
    model = model.to(device)

    checkpoint = torch.load(IMG_CLASSIFIER_PATH)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    train_dataloader = PseudoLabelDataLoader(args, split='train', shuffle=False)
    val_dataloader = PseudoLabelDataLoader(args, split='val', shuffle=False)
    test_dataloader = PseudoLabelDataLoader(args, split='test', shuffle=False)

    device = torch.device('cuda:0')
    n_gpu = 1

    res = {}

    #start inference on train, validation and test set
    with torch.no_grad():
        for batch_idx, (train_batch_id, images, labels) in enumerate(train_dataloader):
            images, labels = (images.to(device), labels.to(device))

            output = model(images)
            predictions = sigmoid(output)

            curr_train_res = np.where(predictions.detach().cpu().numpy() > 0.5, 1, 0)
            res.update(dict(zip(['-'.join(id.split('-')[:2]) for id in train_batch_id], curr_train_res)))

        for batch_idx, (val_batch_id, images, labels) in enumerate(val_dataloader):
            images, labels = (images.to(device), labels.to(device))

            output = model(images)
            predictions = sigmoid(output)

            curr_val_res = np.where(predictions.detach().cpu().numpy() > 0.5, 1, 0)
            res.update(dict(zip(['-'.join(id.split('-')[:2]) for id in val_batch_id], curr_val_res)))

        for batch_idx, (test_batch_id, images, labels) in enumerate(test_dataloader):
            images, labels = (images.to(device), labels.to(device))

            output = model(images)
            predictions = sigmoid(output)

            curr_test_res = np.where(predictions.detach().cpu().numpy() > 0.5, 1, 0)
            res.update(dict(zip(['-'.join(id.split('-')[:2]) for id in test_batch_id], curr_test_res)))

        
        # with open("pseudolabels_14.json", 'w') as f:
        #     json.dump(res, f, indent=2)

        # save it as pickle file
        pickle.dump(res, open("pseudolabels_14.pickle", "wb"))
            

if __name__ == '__main__':
    main()