import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay
from src.models import create_model
# from src.loss_functions.losses import AsymmetricLoss
from src.loss_functions.partial_asymmetric_loss import PartialSelectiveLoss, ComputePrior

from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from src.helper_functions.coco_simulation import simulate_coco
from src.helper_functions.get_data import get_data, get_data_flixstock

import matplotlib.pyplot as plt
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='PyTorch MS_COCO/Flixstock Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset', default='../Datasets/FlixstockTask/')      # '../Datasets/FlixstockTask/'
parser.add_argument('--metadata', type=str, default='./data/COCO_2014')                     # Ignore
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--epochs', default=15, type=int)                           # default is 30
parser.add_argument('--stop_epoch', default=None, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_m')                        # best result used 'tresnet_l'
parser.add_argument('--model-path', default='./pretrained_weights/mtresnet_opim_86.72.pth', type=str)
parser.add_argument('--num-classes', default=21)                                # Flixstock: 21
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--simulate_partial_type', type=str, default=None, help="options are fpc or rps")       # Keep as None
parser.add_argument('--simulate_partial_param', type=float, default=0.5)                                   # Ignore
parser.add_argument('--partial_loss_mode', type=str, default="selective")                    # Selective (CSL) is the best (use at second round)
parser.add_argument('--clip', type=float, default=0)
parser.add_argument('--gamma_pos', type=float, default=0)
parser.add_argument('--gamma_neg', type=float, default=1)
parser.add_argument('--gamma_unann', type=float, default=4)
parser.add_argument('--alpha_pos', type=float, default=1)
parser.add_argument('--alpha_neg', type=float, default=1)
parser.add_argument('--alpha_unann', type=float, default=1)
parser.add_argument('--likelihood_topk', type=int, default=3)                                           # default is 5 (try 3 because neck 6, sleeve_length 3 and pattern 9 is much higher than the others. also, less num_classes so make sense to have less likelihood_topk)
parser.add_argument('--prior_path', type=str, default='./outputs/priors/flixstock_ignore_last_train_avg_preds.csv')      # None if 'ignore' or 'negative' partial_loss_mode is being used (this is to generate the prior for the first time)
parser.add_argument('--prior_threshold', type=float, default=0.5)
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--printfreq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--path_dest', type=str, default="./outputs")
parser.add_argument('--debug_mode', type=str, default="hyperml")


def main():

    # ---------------------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()
    args.do_bottleneck_head = False
    if not os.path.exists(args.path_dest):
        os.makedirs(args.path_dest)

    # ---------------------------------------------------------------------------------
    # Setup model
    print('creating model...')
    model = create_model(args).cuda()
    if args.model_path:  # make sure to load pretrained ImageNet model
        state = torch.load(args.model_path, map_location='cpu')
        try:
            filtered_dict = {k: v for k, v in state['model'].items() if
                            (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        except:
            model.load_state_dict(state, strict=False)

    print('done\n')

    # ---------------------------------------------------------------------------------
    if 'flixstock' in args.data.lower():
        train_loader, val_loader = get_data_flixstock(args)
    else:
        train_loader, val_loader = get_data(args)

    # Actual Training
    train_multi_label_coco(model, train_loader, val_loader, args)


def train_multi_label_coco(model, train_loader, val_loader, args):

    print("Used parameters:")
    print("Image_size:", args.image_size)
    print("Learning_rate:", args.lr)
    print("Epochs:", args.epochs)

    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    print("\nAll classes: ", train_loader.dataset.classes)

    prior = ComputePrior(train_loader.dataset.classes)

    # set optimizer
    Epochs = args.epochs
    if args.stop_epoch is not None:
        Stop_epoch = args.stop_epoch
    else:
        Stop_epoch = args.epochs
    weight_decay = args.weight_decay
    lr = args.lr

    criterion = PartialSelectiveLoss(args)
    # criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=False)

    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):

            inputData = inputData.cuda()
            # target = target.max(dim=1)[0]
            target = target.cuda()
            with autocast():  # mixed precision
                output = model(inputData).float()

            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale > scaler.get_scale())

            if not skip_lr_sched:
                scheduler.step()

            # optimizer.step()

            # scheduler.step()

            ema.update(model)

            prior.update(output)

            # store information
            if i % args.printfreq == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.3f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))

        # Report prior
        prior.save_prior()
        prior.get_top_freq_classes()

        # Save ckpt
        try:
            torch.save(model.state_dict(), os.path.join(args.path_dest, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            print("Model saved successfully.")
        except:
            print("Saving model failed.")

        model.eval()
        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.ckpt'))
            except:
                pass
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))


def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        # target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


if __name__ == '__main__':
    main()
