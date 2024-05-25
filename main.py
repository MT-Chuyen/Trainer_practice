from trainer import trainer

import argparse
import numpy as np
import torch
import random
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-m', '--model', choices=['my_fcn', 'unet', 'deeplab'], default='my_fcn')
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='SegT.Compose([SegT.RandomResizedCrop(64), SegT.ToTensor()])')

    args = parser.parse_args()
    trainer(args)