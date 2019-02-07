import os
import argparse

import torch
import torch.backends.cudnn as cudnn

from loader import BaseTransform
from loader.voc import *
from loader.voc import VOC_CLASSES
from ssd import build_ssd

import pdb


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(description = 'Single Shot MultiBox Detection')

parser.add_argument('--saved', default = 'weights/', help = 'directory for saving checkpoint models')
parser.add_argument('--trained_model', default = 'ssd_300_VOC0712.pth', type = str, help = 'trained model')

parser.add_argument('--save_folder', default = 'eval/', type = str, help = 'folder to save results')
parser.add_argument('--visual_threshold', default = 0.6, type = float, help = 'final confidence threshold')

parser.add_argument('--cuda', default = True, type = str2bool, help = 'use CUDA to train model')

parser.add_argument('--voc_root', default = VOC_ROOT, help = 'location of VOC root directory')

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


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # Dump predictions and assoc. ground truth to text file
    filename = save_folder + 'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i + 1, num_images))
        img = testset.pull_image(i)     # (h, w, c)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)    # Resize and transform to (c, h, w)
        x = x.unsqueeze(0)

        with open(filename, mode = 'a') as f:
            f.write('\nGROUND TRUTH FOR: ' + img_id + '\n')
            for box in annotation:
                f.write('label: ' + ' || '.join(str(b) for b in box) + '\n')

        if cuda:
            x = x.cuda()

        y = net(x)      # Forward pass
        detections = y.data     # (batch, num_classes, top_k, 5); batch is 1; 5 means (score, x1, y1, x2, y2)

        # Scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)): # For each class
            j = 0
            while detections[0, i, j, 0] >= 0.6:    # For each predicted box
                if pred_num == 0:
                    with open(filename, mode = 'a') as f:
                        f.write('PREDICTIONS: ' + '\n')
                score = detections[0, i, j, 0]
                label_name = VOC_CLASSES[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode = 'a') as f:
                    f.write(str(pred_num) + ' label: ' + label_name + ' score: ' +
                            str(score) + ' ' + ' || '.join(str(c) for c in coords) + '\n')
                j += 1


def test_voc():
    # Load ssd net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    ssd_net = build_ssd('test', 300, num_classes) # Initialize SSD
    ssd_net.load_state_dict(torch.load(os.path.join(args.saved, args.trained_model)))
    ssd_net.eval()
    print('Finished loading model!')

    # Load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net).cuda()
        cudnn.benchmark = True

    # Evaluation
    test_net(args.save_folder, ssd_net, args.cuda, testset, BaseTransform(ssd_net.size, (104, 117, 123)), thresh = args.visual_threshold)


if __name__ == '__main__':
    test_voc()