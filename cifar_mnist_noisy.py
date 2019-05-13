from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from cifar_mnist_noisy_module import *
import sys
import pickle
import os.path
import argparse
import logging
from get_fashion_mnist import *

from mxnet.gluon.data.vision import MNIST, FashionMNIST, CIFAR10 
from mxnet.gluon.data.vision import transforms
from gluoncv.data import transforms as gcv_transforms
import copy
import cPickle as pickle

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--batch_size', type=int, help='', default=-1)
parser.add_argument('--epoch_num',  type=int, help='', default=-1)
parser.add_argument('--seed',       type=int, help='', default=1)
parser.add_argument('--model',      type=str, choices=['linear', 'mlp', 'cnn'], default='linear')
parser.add_argument('--cpu',     type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--debug',      type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--fliprule',   type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--mixedmodel', type=str, choices=['yes', 'no'], default='no')
parser.add_argument('--modeltype',  type=str, choices=['regression', 'classification'], default='classification')
parser.add_argument('--good_sample_ratio',  type=float, help='ratio of good samples', default=1.0)
parser.add_argument('--empirical_good_sample_ratio',  type=float, help='empirical ratio of good samples ', default=0.95)
parser.add_argument('--learning_rate',      type=float, help='', default=0.5)
parser.add_argument('--log_file',   type=str, default='log/tmp.log')
parser.add_argument('--loadckpt',   type=str, default='')
parser.add_argument('--saveckptpath',   type=str, default='')
parser.add_argument('--subfix',   type=str, default='')
parser.add_argument('--sub_sampling',      type=float, help='subsampling ratio of the original dataset', default=1.0)
parser.add_argument('--dataset',   type=str, choices=['mnist', 'fashionmnist', 'cifar10', 'cifar100'], default='mnist')
config = parser.parse_args(sys.argv[1:])

def logic_preprocess_config(config):
    
    if config.batch_size == -1:
        logging.warning('batch size is not specified, set to default value 100.')
        config.batch_size = 100

    if config.epoch_num == -1:
        logging.warning('epoch num is not specified, set to default value 10.')
        config.epoch_num = 10
    
    config.cpu      = True if config.cpu == 'yes' else False
    config.debug    = True if config.debug  == 'yes' else False
    config.fliprule     = True if config.fliprule    == 'yes' else False
    config.mixedmodel   = True if config.mixedmodel  == 'yes' else False
    config.regression   = True if config.modeltype == 'regression' else False
    
    assert 0<= config.good_sample_ratio <= 1, 'range of good_sample_ratio is wrong'
    assert 0<= config.empirical_good_sample_ratio <= 1, 'range of empirical good_sample_ratio is wrong'
    
    if config.cpu:
        config.ctx = [mx.cpu(0)]
    else:
        gpus = mx.test_utils.list_gpus()
        config.ctx =  [mx.gpu()] if gpus else [mx.cpu(0)]
    #config.ctx =  [mx.cpu(0), mx.cpu(1)]

    # add seed, fliprule, 
    ckpt_setting_name = 'ds_'       + config.dataset \
                        + '_md_'    + config.model \
                        + '_gr_'    + str(config.good_sample_ratio) \
                        + '_sd_'    + str(config.seed) \
                        + '_ssp_'    + str(config.sub_sampling)
    config.alg_ckpt_name    = config.saveckptpath+'/alg_ckpt_'      + ckpt_setting_name + '_'+config.subfix+'.ckpt'
    return config

config = logic_preprocess_config(config)

logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',  
                    datefmt='%a, %d %b %Y %H:%M:%S',  
                    filename=config.log_file,  
                    filemode='w')  
logging.info(config)

# Fixing the random seed
mx.random.seed(config.seed)

save_data_process = "save_data_process.pickle"

class DataProcess():
    def __init__(self, batch_size, good_sample_ratio, sub_sampling, dataset, mixedmodel=False):

        self.transform_train = transforms.Compose([
            gcv_transforms.RandomCrop(32, pad=4),
            transforms.RandomFlipLeftRight(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        self.dataset_name = dataset 
        self.dataset = {}
        self.label_size = 10
        if dataset == 'mnist':
            dataset_handle = MNIST
        elif dataset == 'fashionmnist':
            dataset_handle = FashionMNIST
        elif dataset == 'cifar10':
            dataset_handle = CIFAR10
        else:
            raise NotImplementedError 

        self.good_sample_ratio = good_sample_ratio
        self.batch_size = batch_size

        if not os.path.isfile(save_data_process):
            print('using handle')
            train = dataset_handle(train=True, )
            test = dataset_handle(train=False, )
            print('finished building train and test')

            size_train = train._label.shape[0]
            print('size of dataset:', size_train)

            if sub_sampling < 1.0:
                size_ptrain = int(size_train * sub_sampling)
                subsample_idx = np.random.choice(size_train, size_ptrain, replace=False)
                train._data = train._data[subsample_idx]
                train._label = train._label[subsample_idx]
            print('finish subsampling')

            self.dataset['train'] = copy.deepcopy(train)
            self.dataset['test'] = copy.deepcopy(test)
            self.dataset['test-bad'] = copy.deepcopy(test)
            
            print('finish deepcopy')
            if self.dataset_name != 'cifar10':
                self.dataset['train']._data = mx.nd.transpose(train._data, axes=(0,3,1,2)).astype('float32')/255.0
                self.dataset['test']._data = mx.nd.transpose(test._data, axes=(0,3,1,2)).astype('float32')/255.0
            
            self.num_train, self.num_test = self.dataset['train']._label.shape[0], self.dataset['test']._label.shape[0]

            
            print('start flipping labels')
            print('making bad training set')
            cnt_label = {}
            for idx, label in enumerate(self.dataset['train']._label):
                cnt_label[label] = cnt_label.get(label, 0) + 1
            cnt_good_label_tgt = {}

            for k, v in cnt_label.items():
                cnt_good_label_tgt[k] = int(v * self.good_sample_ratio)

            manipulate_label = {}
            good_idx_set = []
            for idx, label in enumerate(self.dataset['train']._label):
                manipulate_label[label] = manipulate_label.get(label, 0) + 1
                if manipulate_label[label] > cnt_good_label_tgt[label]:
                    if not mixedmodel:
                        p = np.random.randint(0, self.label_size)
                        while True:
                            if p != label:
                                self.dataset['train']._label[idx] = p
                                break
                            p = np.random.randint(0, self.label_size)
                    else:
                        p = label+1 if label < self.label_size - 1 else 0
                        self.dataset['train']._label[idx] = p
                else:
                    good_idx_set.append(idx)
            self.good_idx_set = good_idx_set

            print('making bad validation set')
            cnt_label_val = {}
            for idx, label in enumerate(self.dataset['test-bad']._label):
                cnt_label_val[label] = cnt_label_val.get(label, 0) + 1
            cnt_good_label_tgt_val = {}

            for k, v in cnt_label_val.items():
                cnt_good_label_tgt_val[k] = int(v * self.good_sample_ratio)

            manipulate_label_val = {}
            good_idx_set_val = []
            for idx, label in enumerate(self.dataset['test-bad']._label):
                manipulate_label_val[label] = manipulate_label_val.get(label, 0) + 1
                if manipulate_label_val[label] > cnt_good_label_tgt_val[label]:
                    if not mixedmodel:
                        p = np.random.randint(0, self.label_size)
                        while True:
                            if p != label:
                                self.dataset['test-bad']._label[idx] = p
                                break
                            p = np.random.randint(0, self.label_size)
                    else:
                        p = label+1 if label < self.label_size - 1 else 0
                        self.dataset['test-bad']._label[idx] = p
                else:
                    good_idx_set_val.append(idx)
            self.good_idx_set_val = good_idx_set_val 

            print('finish flipping labels')
            self.good_idx_array = np.array(self.good_idx_set)
            self.all_idx_array = np.arange(len(self.dataset['train']._label))
            self.bad_idx_array = np.setdiff1d(self.all_idx_array, self.good_idx_array)
            self.dataset['train']._data = mx.nd.concat(self.dataset['train']._data[self.good_idx_array], self.dataset['train']._data[self.bad_idx_array], dim=0)
            self.dataset['train']._label = np.concatenate((self.dataset['train']._label[self.good_idx_array], self.dataset['train']._label[self.bad_idx_array]), axis=0)
            self.good_idx_array = np.arange(len(self.good_idx_array))
            self.bad_idx_array = np.setdiff1d(self.all_idx_array, self.good_idx_array)

            save = {}
            save['train_data'] = self.dataset['train']
            save['test_data'] = self.dataset['test']
            save['test_data_bad'] = self.dataset['test-bad']
            save['good_idx_array'] = self.good_idx_array
            save['bad_idx_array'] = self.bad_idx_array
            save['all_idx_array'] = self.all_idx_array
            with open(save_data_process, "wb") as f:
                pickle.dump(save, f)
        else:
            with open(save_data_process, "rb") as f:
                save = pickle.load(f)
            self.dataset['train'] = save['train_data']
            self.dataset['test'] = save['test_data']
            self.good_idx_array = save['good_idx_array']
            self.bad_idx_array = save['bad_idx_array']
            self.all_idx_array = save['all_idx_array']
            self.dataset['test-bad'] = save['test_data_bad']
        if self.dataset_name == 'cifar10':
            self.val_data = gluon.data.DataLoader(self.dataset['test'].transform_first(self.transform_test), batch_size=batch_size, shuffle=False)
        else:
            self.val_data = gluon.data.DataLoader(self.dataset['test'], batch_size=batch_size, shuffle=False)
        if self.dataset_name == 'cifar10':
            self.val_data_bad = gluon.data.DataLoader(self.dataset['test-bad'].transform_first(self.transform_test), batch_size=batch_size, shuffle=False)
        else:
            self.val_data_bad = gluon.data.DataLoader(self.dataset['test-bad'], batch_size=batch_size, shuffle=False)
        
    def get_data(self, shuffle=True, subset=None):
        if subset is None:
            train_iter = self.get_iter(self.dataset['train'], self.all_idx_array, shuffle=shuffle)
            train_len = len(self.all_idx_array)
        else:
            assert shuffle == True, 'wrong shuffle setting for training'
            train_iter = self.get_iter(self.dataset['train'], subset, shuffle=shuffle)
            train_len = len(subset)
        train_iter_bad = self.get_iter(self.dataset['train'], self.bad_idx_array, shuffle=shuffle )
        return train_iter, self.val_data, train_len, train_iter_bad, self.val_data_bad
    
    def get_iter(self, cur_dataset, good_idx_array, shuffle=True):   
        tmp_dataset = copy.deepcopy(cur_dataset)
        tmp_dataset._data = cur_dataset._data[good_idx_array]
        tmp_dataset._label = cur_dataset._label[good_idx_array]

        if self.dataset_name == 'cifar10':     
            train_data = gluon.data.DataLoader( tmp_dataset.transform_first(self.transform_train),
                                                batch_size=min(self.batch_size, len(good_idx_array)), 
                                                shuffle=shuffle, last_batch='keep')
        else:
            train_data = gluon.data.DataLoader( tmp_dataset,
                                                batch_size=min(self.batch_size, len(good_idx_array)), 
                                                shuffle=shuffle, last_batch='keep')
        return train_data

print('build DataProcess')
D = DataProcess(batch_size        = config.batch_size, 
                good_sample_ratio = config.good_sample_ratio,
                sub_sampling      = config.sub_sampling,
                dataset           = config.dataset,
                mixedmodel        = config.mixedmodel)
print('build TrainClass')
A = Train(epoch               = config.epoch_num,  
            ctx               = config.ctx,
            batch_size        = config.batch_size, 
            lr                = config.learning_rate, 
            regression        = config.regression, 
            model             = config.model, 
            dataset           = config.dataset, 
            debug             = config.debug)

topk_idx = None
if config.loadckpt:
    train_data, val_data, train_len, _, _ = D.get_data(shuffle=False)

    list_v = A.eval(train_data, loadckpt=config.loadckpt)
    print(len(list_v))
    print(train_len, config.empirical_good_sample_ratio)

    k = int(config.empirical_good_sample_ratio * train_len)
    topk_idx = np.argpartition(list_v, k)[:k]
    if config.fliprule:
        topk_idx = np.argpartition(-list_v, k)[:k]

    print(np.sum(topk_idx<=len(D.good_idx_array)), len(topk_idx), k, len(D.good_idx_array))
    logging.info('selected good sample: %d, selected samples: %d, total good samples: %d'%(np.sum(topk_idx<=len(D.good_idx_array)), k, len(D.good_idx_array)))

print('-------- train  --------')
train_data, val_data, _, train_bad, val_data_bad = D.get_data(shuffle=True, subset=topk_idx)

res = A.train(train_data, val_data, val_data_bad, savefilename=config.alg_ckpt_name, train_bad=train_bad)
    
if config.debug:
    print(res)

if config.regression:
    print(min(res))
else:
    print(max(res))
