from util.utils import count_params, set_seed, mIOU
import argparse
from copy import deepcopy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from tqdm import tqdm
from data.dataset import FSSDataset
from model.RanwanNet import LoRA_Sam
from segment_anything_lora import sam_model_registry

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
def parse_args():
    parser = argparse.ArgumentParser(description='IFA for CD-FSS')
    # basic arguments
    parser.add_argument('--data-root', type=str,  default='/home/ranwanwu/CDFSS/dataset', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='fss', choices=['lung', 'deepglobe', 'isic', 'fss'], help='training dataset')
    parser.add_argument('--batch-size', type=int, default=12, help='batch size of training')
    parser.add_argument('--lr', type=float, default=0.004, choices=['fss:0.003', 'deepglobe:0.004', 'isic:0.003', 'lung:0.002'], help='learning rate')
    parser.add_argument('--crop-size', type=int, default=473, help='cropping size of training samples')
    parser.add_argument('--backbone',  type=str, default='resnet101', choices=['resnet50', 'resnet101'], help='backbone of semantic segmentation model')
    parser.add_argument('--shot', type=int, default=5, help='number of support pairs')
    parser.add_argument('--episode', type=int, default=24000, help='total episodes of training')
    parser.add_argument('--snapshot', type=int, default=1200, help='save the model after each snapshot episodes')
    parser.add_argument('--seed', type=int, default=0, help='random seed to generate tesing samples')

    args = parser.parse_args()
    return args

    # lr for 5-shot:[deepglobe:0.004, lung:0.004, isic:0.004, fss:0.004]
    # lr for 1-shot:[deepglobe:0.007, lung:0.003, isic:0.006, fss:0.007]


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

        img_s_list = img_s_list.permute(1, 0, 2, 3, 4)
        mask_s_list = mask_s_list.permute(1, 0, 2, 3)
            
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
                # out_3 = cnn_out_ls[3] * sam_out_ls[3]
                # out_4 = cnn_out_ls[4] * sam_out_ls[4]
                out = 0.5*out_0 + 0.4*out_1 + 0.1*out_2
                # out = 0.3 * out_0 + 0.3 * out_1 + 0.15 * out_2 + 0.15 * out_3 + 0.1 *out_4
                pred = torch.argmax(out, dim=1)
            elif args.dataset == 'lung':
                out_0 = cnn_out_ls[0] * sam_out_ls[0]
                out_1 = cnn_out_ls[1] * sam_out_ls[1]
                out_2 = cnn_out_ls[2] * sam_out_ls[2]
                # out_3 = cnn_out_ls[3] * sam_out_ls[3]
                # out_4 = cnn_out_ls[3] * sam_out_ls[3]                
                out = 0.4*out_0 + 0.4*out_1 + 0.2*out_2
                # out = 0.3 * out_0 + 0.3 * out_1 + 0.15 * out_2 + 0.15 * out_3 + 0.1 *out_4
                pred = torch.argmax(out, dim=1)
            elif args.dataset == 'fss': 
                out = sam_out_ls[2]
                # out_0 = cnn_out_ls[0] * sam_out_ls[0]
                # out_1 = cnn_out_ls[1] * sam_out_ls[1]
                # out_2 = cnn_out_ls[2] * sam_out_ls[2]
                # out_3 = cnn_out_ls[3] * sam_out_ls[3]
                # out_4 = cnn_out_ls[4] * sam_out_ls[4]                
                # out = out_0 + out_1 + out_2
                pred = torch.argmax(out, dim=1)
            elif args.dataset == 'isic':               
                out_0 = cnn_out_ls[0] * sam_out_ls[0]
                out_1 = cnn_out_ls[1] * sam_out_ls[1]
                out_2 = cnn_out_ls[2] * sam_out_ls[2]
                # out_3 = cnn_out_ls[3] * sam_out_ls[3]
                # out_4 = cnn_out_ls[4] * sam_out_ls[4]
                out = 0.1*out_0 + 0.4*out_1 + 0.5*out_2
                # out = 0.1*out_0 + 0.15*out_1 + 0.15*out_2 + 0.3*out_3 + 0.3*out_4
                pred = torch.argmax(out, dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())
        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0

def main():

    args = parse_args()
    print('\n' + str(args))

    if args.dataset == 'deepglobe':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet/deepglobe/train/resnet50_1shot_avg_43.77.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet/deepglobe/train/resnet50_5shot_avg_49.20.pth'
        if args.backbone == 'resnet101':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet_ResNet101/deepglobe/train/resnet101_1shot_avg_43.46.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet_ResNet101/deepglobe/train/resnet101_5shot_avg_49.23.pth'        
        
    if args.dataset == 'isic':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet/isic/train/resnet50_1shot_avg_50.29.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet/isic/train/resnet50_5shot_avg_58.31.pth'
        if args.backbone == 'resnet101':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet_ResNet101/isic/train/resnet101_1shot_avg_45.77.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet_ResNet101/isic/train/resnet101_5shot_avg_65.23.pth'

    if args.dataset == 'lung':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet/lung/train/resnet50_1shot_avg_70.19.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet/lung/train/resnet50_5shot_avg_76.87.pth'
        if args.backbone == 'resnet101':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet_ResNet101/lung/train/resnet101_1shot_avg_57.41.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet_ResNet101/lung/train/resnet101_5shot_avg_60.59.pth'

    if args.dataset == 'fss':
        if args.backbone == 'resnet50':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet/fss/train/resnet50_1shot_avg_78.06.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet/fss/train/resnet50_5shot_avg_79.15.pth'
        if args.backbone == 'resnet101':
            if args.shot == 1:
                checkpoint_path = './outdir/RanwanNet_ResNet101/fss/train/resnet101_1shot_avg_79.31.pth'
            if args.shot == 5:
                checkpoint_path = './outdir/RanwanNet_ResNet101/fss/train/resnet101_5shot_avg_80.32.pth'                
    
    miou = 0
    save_path = 'outdir/RanwanNet_ResNet101/%s/fine_tuning' % (args.dataset)
    os.makedirs(save_path, exist_ok=True)
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    train_dataset = args.dataset + 'ifa'
    trainloader = FSSDataset.build_dataloader(train_dataset, args.batch_size, 4, '0', 'val', args.shot)
    FSSDataset.initialize(img_size=400, datapath=args.data_root)
    testloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, '0', 'val', args.shot)

    sam = sam_model_registry["vit_b"](checkpoint='/home/ranwanwu/CDFSS/pretrained/sam_vit_b_01ec64.pth')
    sam = sam[0]
    model = LoRA_Sam(sam, 4, args.backbone, args.shot).cuda()

    print('Loaded model:', checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    if args.shot == 1:
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True

    else:
        if args.dataset == 'lung':
            for param in model.layer0.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False
            for param in model.layer3.parameters():
                param.requires_grad = False

        else:
            for param in model.layer0.parameters():
                param.requires_grad = False
            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = True
            for param in model.layer3.parameters():
                param.requires_grad = True

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    total_params = 0
    total_trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            total_trainable_params += param.numel()

    print(f"Total number of parameters: {total_params}")
    print(f"Total number of trainable parameters: {total_trainable_params}")

    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)

    model = DataParallel(model).cuda()
    best_model = None

    previous_best = float(miou)
    # each snapshot is considered as an epoch
    for epoch in range(args.episode // args.snapshot):
        
        print("\n==> Epoch %i\t\tlearning rate = %.6f\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0
        tbar = tqdm(trainloader)
        # set_seed(int(time.time()))
        set_seed(0+epoch)

        for i, (img_s_list, mask_s_list, img_q, mask_q, _, _, _) in enumerate(tbar):

            img_s_list = img_s_list.permute(1,0,2,3,4)
            mask_s_list = mask_s_list.permute(1,0,2,3)      
            img_s_list = img_s_list.numpy().tolist()
            mask_s_list = mask_s_list.numpy().tolist()

            img_q, mask_q = img_q.cuda(), mask_q.cuda()

            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = torch.Tensor(img_s_list[k]), torch.Tensor(mask_s_list[k])
                img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

            cnn_out_ls, sam_out_ls = model(img_s_list, mask_s_list, img_q, mask_q)

            if args.dataset == 'deepglobe':
                loss_c0 = 0.7 * criterion(cnn_out_ls[0], mask_q)
                loss_c1 = 0.7 * criterion(cnn_out_ls[1], mask_q)
                loss_c2 = 0.7 * criterion(cnn_out_ls[2], mask_q)
                # loss_c3 = 0.7 * criterion(cnn_out_ls[3], mask_q)
                # loss_c4 = 0.7 * criterion(cnn_out_ls[4], mask_q)
                loss_s0 = 0.3 * criterion(sam_out_ls[0], mask_q)
                loss_s1 = 0.3 * criterion(sam_out_ls[1], mask_q)
                loss_s2 = 0.3 * criterion(sam_out_ls[2], mask_q)
                # loss_s3 = 0.3 * criterion(sam_out_ls[3], mask_q)
                # loss_s4 = 0.3 * criterion(sam_out_ls[4], mask_q)
                loss = loss_c0 + loss_s0 + loss_c1 + loss_s1 + loss_c2 + loss_s2

            elif args.dataset == 'lung':
                loss_c0 = 0.7 * criterion(cnn_out_ls[0], mask_q)
                loss_c1 = 0.7 * criterion(cnn_out_ls[1], mask_q)
                loss_c2 = 0.7 * criterion(cnn_out_ls[2], mask_q)
                # loss_c3 = 0.7 * criterion(cnn_out_ls[3], mask_q)
                # loss_c4 = 0.7 * criterion(cnn_out_ls[4], mask_q)
                loss_s0 = 0.3 * criterion(sam_out_ls[0], mask_q)
                loss_s1 = 0.3 * criterion(sam_out_ls[1], mask_q)
                loss_s2 = 0.3 * criterion(sam_out_ls[2], mask_q)
                # loss_s3 = 0.3 * criterion(sam_out_ls[3], mask_q)
                # loss_s4 = 0.3 * criterion(sam_out_ls[4], mask_q)
                loss = loss_c0 + loss_s0 + loss_c1 + loss_s1 + loss_c2 + loss_s2
         
            elif args.dataset == 'fss':
                loss_c0 = 0.5 * criterion(cnn_out_ls[0], mask_q)
                loss_c1 = 0.5 * criterion(cnn_out_ls[1], mask_q)
                loss_c2 = 0.5 * criterion(cnn_out_ls[2], mask_q)
                # loss_c3 = 0.5 * criterion(cnn_out_ls[3], mask_q)
                # loss_c4 = 0.5 * criterion(cnn_out_ls[4], mask_q)                 
                loss_s0 = 0.5 * criterion(sam_out_ls[0], mask_q)
                loss_s1 = 0.5 * criterion(sam_out_ls[1], mask_q)
                loss_s2 = 0.5 * criterion(sam_out_ls[2], mask_q)
                # loss_s3 = 0.5 * criterion(sam_out_ls[3], mask_q)
                # loss_s4 = 0.5 * criterion(sam_out_ls[4], mask_q)                
                loss = loss_c0 + loss_s0 + loss_c1 + loss_s1 + loss_c2 + loss_s2

            elif args.dataset == 'isic':
                loss_c0 = 0.2 * criterion(cnn_out_ls[0], mask_q)
                loss_c1 = 0.2 * criterion(cnn_out_ls[1], mask_q)
                loss_c2 = 0.2 * criterion(cnn_out_ls[2], mask_q)
                # loss_c3 = 0.2 * criterion(cnn_out_ls[3], mask_q)
                # loss_c4 = 0.2 * criterion(cnn_out_ls[4], mask_q)                
                loss_s0 = 0.8 * criterion(sam_out_ls[0], mask_q)
                loss_s1 = 0.8 * criterion(sam_out_ls[1], mask_q)
                loss_s2 = 0.8 * criterion(sam_out_ls[2], mask_q)
                # loss_s3 = 0.8 * criterion(sam_out_ls[3], mask_q)
                # loss_s4 = 0.8 * criterion(sam_out_ls[4], mask_q)
                loss = loss_c0 + loss_s0 + loss_c1 + loss_s1 + loss_c2 + loss_s2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        model.eval()
        set_seed(args.seed)
        miou = evaluate(model, testloader, args)

        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            torch.save(best_model.module.state_dict(),
                os.path.join(save_path, '%s_%ishot_%.2f.pth' % (args.backbone, args.shot, miou)))
            
    print('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        print('\nRun %i:' % (seed + 1))
        set_seed(args.seed + seed)

        miou = evaluate(best_model, testloader, args)
        total_miou += miou

    print('\n' + '*' * 32)
    print('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    print('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(save_path, '%s_%ishot_avg_%.2f.pth' % (args.backbone, args.shot, total_miou / 5)))

if __name__ == '__main__':
    main()