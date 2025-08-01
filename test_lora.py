from util.utils import count_params, set_seed, mIOU
import argparse
import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import glob
from data.dataset import FSSDataset
from model.MPM import LoRA_Sam
from segment_anything_lora import sam_model_registry

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root', type=str, default='../CDFSS/dataset',  help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='isic', choices=['fss', 'deepglobe', 'isic', 'lung'], help='training dataset')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='backbone of semantic segmentation model')
    parser.add_argument('--shot', type=int, default=1, help='number of support pairs')
    parser.add_argument('--seed', type=int, default=0, help='random seed to generate tesing samples')
    parser.add_argument('--batch-size',  type=int, default=60, help='batch size of training')

    args = parser.parse_args()
    return args


def evaluate(model, dataloader, args):
    tbar = tqdm(dataloader)

    if args.dataset == 'fss':
        num_classes = 1000
    elif args.dataset == 'deepglobe':
        num_classes = 6
    elif args.dataset == 'isic':
        num_classes = 3
    elif args.dataset == 'lung':
        num_classes = 1

    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):

        img_s_list = img_s_list.permute(1,0,2,3,4)
        mask_s_list = mask_s_list.permute(1,0,2,3)
            
        img_s_list = img_s_list.numpy().tolist()
        mask_s_list = mask_s_list.numpy().tolist()

        img_q, mask_q = img_q.cuda(), mask_q.cuda()

        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

        cls = cls[0].item()
        cls = cls + 1

        with torch.no_grad():
            cnn_out_ls, sam_out_ls = model(img_s_list, mask_s_list, img_q, mask_q)
            if args.dataset == 'deepglobe':
                out_0 = cnn_out_ls[0] * sam_out_ls[0]
                out_1 = cnn_out_ls[1] * sam_out_ls[1]
                out_2 = cnn_out_ls[2] * sam_out_ls[2]
                out = 0.5*out_0 + 0.4*out_1 + 0.1*out_2
                pred = torch.argmax(out, dim=1)
            elif args.dataset == 'lung':
                out_0 = cnn_out_ls[0] * sam_out_ls[0]
                out_1 = cnn_out_ls[1] * sam_out_ls[1]
                out_2 = cnn_out_ls[2] * sam_out_ls[2]
                out = 0.4*out_0 + 0.4*out_1 + 0.2*out_2
                pred = torch.argmax(out, dim=1)
            elif args.dataset == 'fss':
                out_0 = cnn_out_ls[0] * sam_out_ls[0]
                out_1 = cnn_out_ls[1] * sam_out_ls[1]
                out_2 = cnn_out_ls[2] * sam_out_ls[2]
                out = out_0 + out_1 + out_2
                pred = torch.argmax(out, dim=1)
            elif args.dataset == 'isic':
                out_0 = cnn_out_ls[0] * sam_out_ls[0]
                out_1 = cnn_out_ls[1] * sam_out_ls[1]
                out_2 = cnn_out_ls[2] * sam_out_ls[2]
                out = 0.1*out_0 + 0.4*out_1 + 0.5*out_2
                pred = torch.argmax(out, dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():
    args = parse_args()
    print('\n' + str(args))

    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    sam = sam_model_registry["vit_b"](checkpoint='../CDFSS/pretrained/sam_vit_b_01ec64.pth')  # "sam_vit_b_01ec64.pth")
    sam = sam[0]
    model = LoRA_Sam(sam, 4, args.backbone, args.shot).cuda()

    ### Please modify the following paths with your model path if needed.
    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './deepglobe/fine_tuning/resnet50_1shot_avg_50.50.pth'
            if args.shot == 5:
                checkpoint_path = './deepglobe/fine_tuning/resnet50_5shot_avg_58.61.pth'
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './isic/fine_tuning/resnet50_1shot_avg_74.38.pth'
            if args.shot == 5:
                checkpoint_path = './isic/fine_tuning/resnet50_5shot_avg_77.93.pth'
    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './lung/fine_tuning/resnet50_1shot_avg_76.91.pth'
            if args.shot == 5:
                checkpoint_path = './lung/fine_tuning/resnet50_5shot_avg_78.19.pth'
    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './fss/fine_tuning/resnet50_1shot_avg_83.22.pth'
            if args.shot == 5:
                checkpoint_path = './fss/fine_tuning/resnet50_5shot_avg_84.89.pth'

    print('Evaluating model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    print('\nParams: %.1fM' % count_params(model))

    # best_model = DataParallel(model).cuda()
    best_model = model.cuda()

    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    model.eval()
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

if __name__ == '__main__':
    main()
