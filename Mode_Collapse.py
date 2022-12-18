
import torch
from modules.dataloaders import R2DataLoader
from modules.tokenizers import Tokenizer
import argparse
import json
from models.models import XProNet

import pdb


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='data/iu_xray/labels.pickle',
                        help='the path to the directory containing the data.')

    parser.add_argument('--img_init_protypes_path', type=str, default='data/iu_xray/init_protypes_512.pt',
                        help='the path to the directory containing the data.')
    parser.add_argument('--init_protypes_path', type=str, default='data/iu_xray/init_protypes_512.pt',
                        help='the path to the directory containing the data.')

    parser.add_argument('--text_init_protypes_path', type=str, default='data/iu_xray/text_empty_initprotypes_512.pt',
                        help='the path to the directory containing the data.')
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_txt_ebd', type=int, default=768, help='the dimension of extracted text embedding.')
    parser.add_argument('--d_img_ebd', type=int, default=512, help='the dimension of extracted img embedding.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

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

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--img_num_protype', type=int, default=10, help='.')
    parser.add_argument('--text_num_protype', type=int, default=10, help='.')
    parser.add_argument('--gbl_num_protype', type=int, default=10, help='.')
    parser.add_argument('--num_protype', type=int, default=10, help='.')
    parser.add_argument('--num_cluster', type=int, default=20, help='.')
    parser.add_argument('--weight_img_con_loss', type=float, default=1, help='.')
    parser.add_argument('--weight_txt_con_loss', type=float, default=1, help='.')

    parser.add_argument('--weight_img_bce_loss', type=float, default=1, help='.')
    parser.add_argument('--weight_txt_bce_loss', type=float, default=1, help='.')
    parser.add_argument('--img_con_margin', type=float, default=0.4, help='.')
    parser.add_argument('--txt_con_margin', type=float, default=0.4, help='.')

    args = parser.parse_args()
    return args


def main():

    args = parse_agrs()

    # create tokenizer
    tokenizer = Tokenizer(args)

    # build model architecture
    model = XProNet(args, tokenizer)

    #set path
    PATH = "results/iu_xray/iu_xray.pth"

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

    checkpoint = torch.load(PATH)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    device = torch.device('cuda:0')
    n_gpu = 1

    #start inference on validation and test set
    with torch.no_grad():
        val_gts, val_res, val_lab = [], [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(val_dataloader):
            images, reports_ids, reports_masks, labels = images.to(device), reports_ids.to(
                device), reports_masks.to(device), labels.to(device)

            output, _ = model(images, labels = labels, mode='sample')
            # change to model.module for multi-gpu
            if n_gpu>1:
                reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            else:
                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            val_res.extend(reports)
            val_gts.extend(ground_truths)
            val_lab.extend(labels.cpu().numpy().tolist())

        test_gts, test_res, test_lab = [], [], []
        example_images = []
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(test_dataloader):
            images, reports_ids, reports_masks, labels = images.to(device), reports_ids.to(
                device), reports_masks.to(device), labels.to(device)
            output, _ = model(images, labels=labels, mode='sample')
            if n_gpu>1:
                reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            else:
                reports = model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)
            test_lab.extend(labels.cpu().numpy().tolist())

    with open("val_res_temp.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(val_res, f, indent=2)

    with open("val_gts_temp.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(val_gts, f, indent=2)

    with open("val_lab_temp.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(val_lab, f, indent=2)

    with open("test_res_temp.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(test_res, f, indent=2)

    with open("test_gts_temp.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(test_gts, f, indent=2)

    with open("test_lab_temp.json", 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(test_lab, f, indent=2)

if __name__ == '__main__':
    main()






