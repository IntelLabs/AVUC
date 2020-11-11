import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.bayesian.resnet as resnet
import models.deterministic.resnet as det_resnet
import numpy as np
from src import util
#from utils import calib
import csv
from src.util import get_rho
from src.avuc_loss import AvULoss

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet20',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet20)')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=512,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq',
                    '-p',
                    default=50,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 20)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained',
                    dest='pretrained',
                    action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half',
                    dest='half',
                    action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir',
                    dest='save_dir',
                    help='The directory used to save the trained models',
                    default='./checkpoint/bayesian_svi_avu',
                    type=str)
parser.add_argument(
    '--save-every',
    dest='save_every',
    help='Saves checkpoints at every specified number of epochs',
    type=int,
    default=10)
parser.add_argument('--mode', type=str, required=True, help='train | test')
parser.add_argument('--num_monte_carlo',
                    type=int,
                    default=20,
                    metavar='N',
                    help='number of Monte Carlo samples')
parser.add_argument(
    '--tensorboard',
    type=bool,
    default=True,
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument('--val_batch_size', default=10000, type=int)
parser.add_argument(
    '--log_dir',
    type=str,
    default='./logs/cifar/bayesian_svi_avu',
    metavar='N',
    help='use tensorboard for logging and visualization of training progress')
parser.add_argument(
    '--moped',
    type=bool,
    default=False,
    help='set prior and initialize approx posterior with Empirical Bayes')
parser.add_argument('--delta',
                    type=float,
                    default=1.0,
                    help='delta value for variance scaling in MOPED')
optimal_threshold = 1.0
opt_th = optimal_threshold
best_prec1 = 0
best_avu = 0
len_trainset = 50000
len_testset = 10000
beta = 3.0


class CorruptDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class OODDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


def get_ood_dataloader(ood_images, ood_labels):
    ood_dataset = OODDataset(ood_images,
                             ood_labels,
                             transform=transforms.Compose(
                                 [transforms.ToTensor()]))
    ood_data_loader = torch.utils.data.DataLoader(ood_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    return ood_data_loader


corruptions = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
    'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'pixelate', 'saturate', 'shot_noise', 'spatter', 'speckle_noise',
    'zoom_blur'
]


def get_corrupt_dataloader(corrupted_images, corrupted_labels, level):

    corrupted_images_1 = corrupted_images[0:10000, :, :, :]
    corrupted_labels_1 = corrupted_labels[0:10000]
    corrupted_images_2 = corrupted_images[10000:20000, :, :, :]
    corrupted_labels_2 = corrupted_labels[10000:20000]
    corrupted_images_3 = corrupted_images[20000:30000, :, :, :]
    corrupted_labels_3 = corrupted_labels[20000:30000]
    corrupted_images_4 = corrupted_images[30000:40000, :, :, :]
    corrupted_labels_4 = corrupted_labels[30000:40000]
    corrupted_images_5 = corrupted_images[40000:50000, :, :, :]
    corrupted_labels_5 = corrupted_labels[40000:50000]

    if level == 1:
        corrupt_val_dataset = CorruptDataset(corrupted_images_1,
                                             corrupted_labels_1,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor()]))
    elif level == 2:
        corrupt_val_dataset = CorruptDataset(corrupted_images_2,
                                             corrupted_labels_2,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor()]))
    elif level == 3:
        corrupt_val_dataset = CorruptDataset(corrupted_images_3,
                                             corrupted_labels_3,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor()]))
    elif level == 4:
        corrupt_val_dataset = CorruptDataset(corrupted_images_4,
                                             corrupted_labels_4,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor()]))
    elif level == 5:
        corrupt_val_dataset = CorruptDataset(corrupted_images_5,
                                             corrupted_labels_5,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor()]))

    corrupt_val_loader = torch.utils.data.DataLoader(
        corrupt_val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    return corrupt_val_loader


def MOPED_layer(layer, det_layer, delta):
    """
    Set the priors and initialize surrogate posteriors of Bayesian NN with Empirical Bayes
    MOPED (Model Priors with Empirical Bayes using Deterministic DNN)
    Ref: https://arxiv.org/abs/1906.05323
         'Specifying Weight Priors in Bayesian Deep Neural Networks with Empirical Bayes'. AAAI 2020.
    """
    if (str(layer) == 'Conv2dVariational()'):
        #set the priors
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize surrogate posteriors
        layer.mu_kernel.data = det_layer.weight.data
        #layer.rho_kernel.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            #layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif (isinstance(layer, nn.Conv2d)):
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data2

    elif (str(layer) == 'LinearVariational()'):
        print(str(layer))
        layer.prior_weight_mu = det_layer.weight.data
        if layer.prior_bias_mu is not None:
            layer.prior_bias_mu = det_layer.bias.data

        #initialize the surrogate posteriors
        layer.mu_weight.data = det_layer.weight.data
        layer.rho_weight.data = get_rho(det_layer.weight.data, delta)
        if layer.mu_bias is not None:
            layer.mu_bias.data = det_layer.bias.data
            layer.rho_bias.data = get_rho(det_layer.bias.data, delta)

    elif str(layer).startswith('Batch'):
        #initialize parameters
        print(str(layer))
        layer.weight.data = det_layer.weight.data
        if layer.bias is not None:
            layer.bias.data = det_layer.bias.data
        layer.running_mean.data = det_layer.running_mean.data
        layer.running_var.data = det_layer.running_var.data
        layer.num_batches_tracked.data = det_layer.num_batches_tracked.data


def main():
    global args, best_prec1, best_avu
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    tb_writer = None
    if args.tensorboard:
        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tb_writer = SummaryWriter(logger_dir)

    preds_dir = os.path.join(args.log_dir, 'preds')
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
    results_dir = os.path.join(args.log_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()
        ]),
        download=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        root='./data',
        train=False,
        transform=transforms.Compose([transforms.ToTensor()])),
                                             batch_size=args.val_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
        avu_criterion = AvULoss().cuda()
    else:    
        criterion = nn.CrossEntropyLoss().cpu()
        avu_criterion = AvULoss().cpu()

    if args.half:
        model.half()
        criterion.half()

    if args.arch in ['resnet110']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.mode == 'train':

        if (args.moped):
            print("MOPED enabled")
            det_model = torch.nn.DataParallel(det_resnet.__dict__[args.arch]())
            if torch.cuda.is_available():
                det_model.cuda()
            else:
                det_model.cpu()
            checkpoint_file = 'checkpoint/deterministic/{}_cifar.pth'.format(
                args.arch)
            checkpoint = torch.load(checkpoint_file)
            det_model.load_state_dict(checkpoint['state_dict'])

            for (idx_1, layer_1), (det_idx_1, det_layer_1) in zip(
                    enumerate(model.children()),
                    enumerate(det_model.children())):
                MOPED_layer(layer_1, det_layer_1, args.delta)
                for (idx_2, layer_2), (det_idx_2, det_layer_2) in zip(
                        enumerate(layer_1.children()),
                        enumerate(det_layer_1.children())):
                    MOPED_layer(layer_2, det_layer_2, args.delta)
                    for (idx_3, layer_3), (det_idx_3, det_layer_3) in zip(
                            enumerate(layer_2.children()),
                            enumerate(det_layer_2.children())):
                        MOPED_layer(layer_3, det_layer_3, args.delta)
                        for (idx_4, layer_4), (det_idx_4, det_layer_4) in zip(
                                enumerate(layer_3.children()),
                                enumerate(det_layer_3.children())):
                            MOPED_layer(layer_4, det_layer_4, args.delta)

        model.state_dict()

        for epoch in range(args.start_epoch, args.epochs):

            lr = args.lr
            if (epoch >= 80 and epoch < 120):
                lr = 0.1 * args.lr
            elif (epoch >= 120 and epoch < 160):
                lr = 0.01 * args.lr
            elif (epoch >= 160 and epoch < 180):
                lr = 0.001 * args.lr
            elif (epoch >= 180):
                lr = 0.0005 * args.lr

            optimizer = torch.optim.Adam(model.parameters(), lr)
            # train for one epoch
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
            train(train_loader, model, criterion, avu_criterion, optimizer,
                  epoch, tb_writer)

            prec1 = validate(val_loader, model, criterion, avu_criterion,
                             epoch, tb_writer)

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if epoch > 0:
                if is_best:
                    save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_prec1': best_prec1,
                        },
                        is_best,
                        filename=os.path.join(
                            args.save_dir,
                            'bayesian_{}_cifar.pth'.format(args.arch)))

    elif args.mode == 'test':
        checkpoint_file = args.save_dir + '/bayesian_{}_cifar.pth'.format(
            args.arch)
        if torch.cuda.is_available(): 
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        print('load checkpoint. epoch :', checkpoint['epoch'])
        model.load_state_dict(checkpoint['state_dict'])

        #header = ['corrupt', 'test_acc', 'brier', 'ece']
        header = ['corrupt', 'test_acc']

        #Evaluate on OOD dataset (SVHN)
        ood_images_file = 'data/SVHN/svhn-test.npy'
        ood_images = np.load(ood_images_file)
        ood_images = ood_images[:10000, :, :, :]
        ood_labels = np.arange(len(ood_images)) + 10  #create dummy labels
        ood_loader = get_ood_dataloader(ood_images, ood_labels)
        ood_acc = evaluate(model, ood_loader, corrupt='ood', level=None)
        print('******OOD data***********\n')
        #print('ood_acc: ', ood_acc, ' | Brier: ', ood_brier, ' | ECE: ', ood_ece, '\n')
        print('ood_acc: ', ood_acc)
        '''
        o_file = args.log_dir + '/results/ood_results.csv'
        with open(o_file, 'wt') as o_file:
            writer = csv.writer(o_file, delimiter=',', lineterminator='\n')
            writer.writerow([j for j in header])
            writer.writerow(['ood', ood_acc, ood_brier, ood_ece])
        o_file.close() 
        '''
        #Evaluate on test dataset
        test_acc = evaluate(model, val_loader, corrupt=None, level=None)
        print('******Test data***********\n')
        #print('test_acc: ', test_acc, ' | Brier: ', brier, ' | ECE: ', ece, '\n')
        print('test_acc: ', test_acc)
        '''
        t_file = args.log_dir + '/results/test_results.csv'
        with open(t_file, 'wt') as t_file:
            writer = csv.writer(t_file, delimiter=',', lineterminator='\n')
            writer.writerow([j for j in header])
            writer.writerow(['test', test_acc, brier, ece])
        t_file.close()
        '''

        for level in range(1, 6):
            print('******Corruption Level: ', level, ' ***********\n')
            results_file = args.log_dir + '/results/level' + str(
                level) + '.csv'
            with open(results_file, 'wt') as results_file:
                writer = csv.writer(results_file,
                                    delimiter=',',
                                    lineterminator='\n')
                writer.writerow([j for j in header])
                for c in corruptions:
                    images_file = 'data/CIFAR-10-C/' + c + '.npy'
                    labels_file = 'data/CIFAR-10-C/labels.npy'
                    corrupt_images = np.load(images_file)
                    corrupt_labels = np.load(labels_file)

                    val_loader = get_corrupt_dataloader(corrupt_images,
                                                        corrupt_labels,
                                                        level=level)
                    test_acc = evaluate(model,
                                        val_loader,
                                        corrupt=c,
                                        level=level)
                    print('############ Corruption type: ', c,
                          ' ################')
                    #print('test_acc: ', test_acc, ' | Brier: ', brier, ' | ECE: ', ece, '\n')
                    print('test_acc: ', test_acc)
                    #writer.writerow([c, test_acc, brier, ece])
                    writer.writerow([c, test_acc])
            results_file.close()


def train(train_loader,
          model,
          criterion,
          avu_criterion,
          optimizer,
          epoch,
          tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    avg_unc = AverageMeter()

    # switch to train mode
    model.train()

    preds_list = []
    labels_list = []
    unc_list = []
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
        else:
            target = target.cpu()
            input_var = input.cpu()
            target_var = target.cpu()
        if args.half:
            input_var = input_var.half()

        optimizer.zero_grad()

        output, kl = model(input_var)
        probs_ = torch.nn.functional.softmax(output, dim=1)
        probs = probs_.data.cpu().numpy()

        pred_entropy = util.entropy(probs)
        unc = np.mean(pred_entropy, axis=0)
        preds = np.argmax(probs, axis=-1)
        preds_list.append(preds)
        labels_list.append(target.cpu().data.numpy())
        unc_list.append(pred_entropy)

        AvU = util.accuracy_vs_uncertainty(np.array(preds),
                                           np.array(target.cpu().data.numpy()),
                                           np.array(pred_entropy), opt_th)

        cross_entropy_loss = criterion(output, target_var)
        scaled_kl = kl.data / len_trainset
        elbo_loss = cross_entropy_loss + scaled_kl
        avu_loss = beta * avu_criterion(output, target_var, opt_th, type=0)
        loss = cross_entropy_loss + scaled_kl + avu_loss

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        avg_unc.update(unc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Avg_Unc {avg_unc.val:.3f} ({avg_unc.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      avg_unc=avg_unc))

        if tb_writer is not None:
            tb_writer.add_scalar('train/cross_entropy_loss',
                                 cross_entropy_loss.item(), epoch)
            tb_writer.add_scalar('train/kl_div', scaled_kl.item(), epoch)
            tb_writer.add_scalar('train/elbo_loss', elbo_loss.item(), epoch)
            tb_writer.add_scalar('train/avu_loss', avu_loss, epoch)
            tb_writer.add_scalar('train/loss', loss.item(), epoch)
            tb_writer.add_scalar('train/AvU', AvU, epoch)
            tb_writer.add_scalar('train/accuracy', prec1.item(), epoch)
            tb_writer.flush()

    print('TRAINING EPOCH: ', epoch)
    preds = np.hstack(np.asarray(preds_list))
    labels = np.hstack(np.asarray(labels_list))
    unc_ = np.hstack(np.asarray(unc_list))
    unc_correct = np.take(unc_, np.where(preds == labels))
    unc_incorrect = np.take(unc_, np.where(preds != labels))
    print('avg unc correct preds: ',
          np.mean(np.take(unc_, np.where(preds == labels)), axis=1))
    print('avg unc incorrect preds: ',
          np.mean(np.take(unc_, np.where(preds != labels)), axis=1))


def validate(val_loader,
             model,
             criterion,
             avu_criterion,
             epoch,
             tb_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    avg_unc = AverageMeter()
    global opt_th

    # switch to evaluate mode
    model.eval()

    end = time.time()
    preds_list = []
    labels_list = []
    unc_list = []
    th_list = np.linspace(0, 1, 21)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                input_var = input.cuda()
                target_var = target.cuda()
            else:
                target = target.cpu()
                input_var = input.cpu()
                target_var = target.cpu()

            if args.half:
                input_var = input_var.half()

            output, kl = model(input_var)
            probs_ = torch.nn.functional.softmax(output, dim=1)
            probs = probs_.data.cpu().numpy()

            pred_entropy = util.entropy(probs)
            unc = np.mean(pred_entropy, axis=0)
            preds = np.argmax(probs, axis=-1)
            preds_list.append(preds)
            labels_list.append(target.cpu().data.numpy())
            unc_list.append(pred_entropy)

            AvU = util.accuracy_vs_uncertainty(
                np.array(preds), np.array(target.cpu().data.numpy()),
                np.array(pred_entropy), opt_th)

            cross_entropy_loss = criterion(output, target_var)
            scaled_kl = kl.data / len_trainset
            elbo_loss = cross_entropy_loss + scaled_kl
            avu_loss = beta * avu_criterion(output, target_var, opt_th, type=0)
            loss = cross_entropy_loss + scaled_kl + avu_loss

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            avg_unc.update(unc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Avg_Unc {avg_unc.val:.3f} ({avg_unc.avg:.3f})'.format(
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          loss=losses,
                          top1=top1,
                          avg_unc=avg_unc))

            if tb_writer is not None:
                tb_writer.add_scalar('val/cross_entropy_loss',
                                     cross_entropy_loss.item(), epoch)
                tb_writer.add_scalar('val/kl_div', scaled_kl.item(), epoch)
                tb_writer.add_scalar('val/elbo_loss', elbo_loss.item(), epoch)
                tb_writer.add_scalar('val/avu_loss', avu_loss, epoch)
                tb_writer.add_scalar('val/loss', loss.item(), epoch)
                tb_writer.add_scalar('val/AvU', AvU, epoch)
                tb_writer.add_scalar('val/accuracy', prec1.item(), epoch)
                tb_writer.flush()

        preds = np.hstack(np.asarray(preds_list))
        labels = np.hstack(np.asarray(labels_list))
        unc_ = np.hstack(np.asarray(unc_list))
        avu_th, unc_th = util.eval_avu(preds, labels, unc_)
        print('max AvU: ', np.amax(avu_th))
        unc_correct = np.take(unc_, np.where(preds == labels))
        unc_incorrect = np.take(unc_, np.where(preds != labels))
        print('avg unc correct preds: ',
              np.mean(np.take(unc_, np.where(preds == labels)), axis=1))
        print('avg unc incorrect preds: ',
              np.mean(np.take(unc_, np.where(preds != labels)), axis=1))
        '''
        print('unc @max AvU: ', unc_th[np.argmax(avu_th)])
        print('avg unc: ', np.mean(unc_, axis=0))
        print('avg unc: ', np.mean(unc_th, axis=0))
        print('min unc: ', np.amin(unc_))
        print('max unc: ', np.amax(unc_))
        '''
        if epoch <= 5:
            opt_th = (np.mean(unc_correct, axis=1) +
                      np.mean(unc_incorrect, axis=1)) / 2

    print('opt_th: ', opt_th)
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def evaluate(model, val_loader, corrupt=None, level=None):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pred_probs_mc = []
        for batch_idx, (data, target) in enumerate(val_loader):
            #print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            for mc_run in range(args.num_monte_carlo):
                model.eval()
                output, _ = model.forward(data)
                pred_probs = torch.nn.functional.softmax(output, dim=1)
                pred_probs_mc.append(pred_probs.cpu().data.numpy())

        if corrupt == 'ood':
            np.save(args.log_dir + '/preds/svi_avu_ood_probs.npy',
                    pred_probs_mc)
            print('saved predictions')
            return None

        target_labels = target.cpu().data.numpy()
        pred_mean = np.mean(pred_probs_mc, axis=0)
        Y_pred = np.argmax(pred_mean, axis=1)
        test_acc = (Y_pred == target_labels).mean()
        #brier = np.mean(calib.brier_scores(target_labels, probs=pred_mean))
        #ece = calib.expected_calibration_error_multiclass(pred_mean, target_labels)
        print('Test accuracy:', test_acc * 100)
        #print('Brier score: ', brier)
        #print('ECE: ', ece)
        if corrupt is not None:
            np.save(
                args.log_dir +
                '/preds/svi_avu_corrupt-static-{}-{}_probs.npy'.format(
                    corrupt, level), pred_probs_mc)
            np.save(
                args.log_dir +
                '/preds/svi_avu_corrupt-static-{}-{}_labels.npy'.format(
                    corrupt, level), target_labels)
            print('saved predictions')
        elif corrupt == 'test':
            np.save(args.log_dir + '/preds/svi_avu_test_probs.npy',
                    pred_probs_mc)
            np.save(args.log_dir + '/preds/svi_avu_test_labels.npy',
                    target_labels)
            print('saved predictions')
    return test_acc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
