#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train the DTLN model in default settings. The folders for noisy and
clean files are expected to have the same number of files and the files to 
have the same name. The training procedure always saves the best weights of 
the model into the folder "./models_'runName'/". Also a log file of the 
training progress is written there. To change any parameters go to the 
"DTLN_model.py" file or use "modelTrainer.parameter = XY" in this file.
It is recommended to run the training on a GPU. The setup is optimized for the
DNS-Challenge data set. If you use a custom data set, just play around with
the parameters.

Example call:
    $python run_training.py --train_mix /path/to/noisy/train \
                           --train_speech /path/to/clean/train \
                           --val_mix /path/to/noisy/val \
                           --val_speech /path/to/clean/val \
                           --run_name DTLN_model \
                           --gpu 0

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: Updated with argparse support

This code is licensed under the terms of the MIT-license.
"""

from DTLN_model import DTLN_model
import os
import argparse


if __name__ == '__main__':
    # argument parser for running from command line
    parser = argparse.ArgumentParser(description='Train DTLN model')
    parser.add_argument('--train_mix', '-tm', required=True,
                        help='path to folder containing noisy/mixed training audio files')
    parser.add_argument('--train_speech', '-ts', required=True,
                        help='path to folder containing clean/speech training audio files')
    parser.add_argument('--val_mix', '-vm', required=True,
                        help='path to folder containing noisy/mixed validation audio files')
    parser.add_argument('--val_speech', '-vs', required=True,
                        help='path to folder containing clean/speech validation audio files')
    parser.add_argument('--run_name', '-r', default='DTLN_model',
                        help='name for this training run (default: DTLN_model)')
    parser.add_argument('--gpu', '-g', default='0',
                        help='GPU device index to use (default: 0)')
    parser.add_argument('--deterministic', '-d', action='store_true',
                        help='enable deterministic operations for reproducibility')
    
    args = parser.parse_args()
    
    # use the specified GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # activate this for some reproducibility
    if args.deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # path to folders from arguments
    path_to_train_mix = args.train_mix
    path_to_train_speech = args.train_speech
    path_to_val_mix = args.val_mix
    path_to_val_speech = args.val_speech
    runName = args.run_name

    # create instance of the DTLN model class
    modelTrainer = DTLN_model()
    # build the model
    modelTrainer.build_DTLN_model()
    # compile it with optimizer and cost function for training
    modelTrainer.compile_model()
    # train the model
    modelTrainer.train_model(runName, path_to_train_mix, path_to_train_speech, \
                             path_to_val_mix, path_to_val_speech)



