import os
import time
import argparse

import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.backends.cudnn as cudnn

from loader.voc import *
from loader.config import voc, MEANS
from layers.modules.multibox_loss import MultiBoxLoss
from utils.data_aug import SSDAugmentation
from ssd import build_ssd

import pdb


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(description = 'Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--dataset', default = 'VOC', choices = ['VOC'], type = str, help = 'currently only support VOC2007')
parser.add_argument('--dataset_root', default = VOC_ROOT, help = 'dataset root directory path')
parser.add_argument('--basenet', default = 'vgg16_reducedfc.pth', help = 'pretrained base model')

parser.add_argument('--resume', default = None, type = str, help = 'checkpoint state_dict file to resume training from')

parser.add_argument('--start_iter', default = 0, type = int, help = 'resume training at this iter')
parser.add_argument('--batch_size', default = 32, type = int, help = 'batch size for training')
parser.add_argument('--num_workers', default = 4, type = int, help = 'number of workers used in dataloading')

parser.add_argument('--lr', default = 1e-3, type = float, help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, help = 'momentum value for optim')
parser.add_argument('--weight_decay', default = 5e-4, type = float, help = 'weight decay for SGD')
parser.add_argument('--gamma', default = 0.1, type = float, help = 'gamma update for SGD')

parser.add_argument('--saved', default = 'weights/', help = 'directory for saving checkpoint models')

parser.add_argument('--cuda', default = True, type = str2bool, help = 'use CUDA to train model')
parser.add_argument('--visdom', default = True, type = str2bool, help = 'use visdom for loss visualization')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print('WARNING: It looks like you have a CUDA device, but aren\'t using CUDA.\n'
              'Run with --cuda for optimal training speed.')
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


if not os.path.exists(args.saved):
    os.mkdir(args.saved)


if args.visdom:
    import visdom
    viz = visdom.Visdom()


def train():

    cfg, dataset = None, None
    if args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(root = args.dataset_root, transform = SSDAugmentation(cfg['min_dim'], MEANS))
    else:
        raise Exception('No such dataset {} supported yet!'.format(args.dataset))

    data_loader = data.DataLoader(dataset, args.batch_size, num_workers = args.num_workers, shuffle = True,
                                  collate_fn = detection_collate, pin_memory = True)

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net).cuda()
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.saved + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        # Initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)


    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()

    # Loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)


    # Create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
            # Reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # Learning rate decay
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # Load train data
        images, targets = next(batch_iterator)  # (batch_size, 3, h, w); a list of length batch_size

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

        t0 = time.time()

        # Forward pass
        out = net(images)

        # Backward pass
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        t1 = time.time()

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end = ' ')

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(), iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC2007_' + repr(iteration) + '.pth')

        torch.save(ssd_net.state_dict(), args.saved + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type, epoch_size = 1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # Initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()