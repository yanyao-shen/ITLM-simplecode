from __future__ import print_function
import numpy as np
import mxnet as mx
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import logging
from gluoncv import model_zoo
from gluoncv.model_zoo.cifarwideresnet import CIFARWideResNet, CIFARBasicBlockV2
import os
def get_cifar100_wide_resnet(num_layers, width_factor=1, drop_rate=0.0,
                          pretrained=False, 
                          root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 6*n+2, e.g. 20, 56, 110, 164.
    width_factor: int
        The width factor to apply to the number of channels from the original resnet.
    drop_rate: float
        The rate of dropout.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert (num_layers - 4) % 6 == 0

    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [16, 16*width_factor, 32*width_factor, 64*width_factor]

    net = CIFARWideResNet(CIFARBasicBlockV2, layers, channels, drop_rate, classes=100, **kwargs)
    return net


import mxnet.ndarray as F

class NetMNIST(gluon.Block):
    def __init__(self, **kwargs):
        super(NetMNIST, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
            self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
            self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.fc1 = nn.Dense(500)
            self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

class NetCIFAR100(gluon.Block):
    def __init__(self, **kwargs):
        super(NetMNIST, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
            self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
            self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides = (2,2))
            self.fc1 = nn.Dense(500)
            self.fc2 = nn.Dense(100)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x

class Train():
    def __init__(self, epoch,  ctx, 
                        batch_size,  lr, regression, 
                        model, dataset, debug=False):
        self.epoch = epoch

        self.ctx = ctx 
        self.batch_size = batch_size
        self.print_once = False
        self.regression = regression
        self.model = model
        self.debug = debug
        self.dataset = dataset 


        # define network
        
        if self.model == 'linear':
            net = nn.Sequential()
            with net.name_scope():
                if self.regression:
                    net.add(nn.Dense(1))
                else:
                    if dataset in ['mnist', 'fashionmnist', 'cifar10']:
                        net.add(nn.Dense(10))
                    elif dataset == 'cifar100':
                        net.add(nn.Dense(100))
                    else:
                        raise NotImplementedError
        elif self.model == 'mlp':
            if dataset == 'mnist':
                net = nn.Sequential()
                with net.name_scope():
                    net.add(nn.Dense(128, activation='relu'))
                    net.add(nn.Dense(64, activation='relu'))
                    if self.regression:
                        net.add(nn.Dense(1))
                    else:
                        net.add(nn.Dense(10))
            elif dataset in ['fashionmnist', 'cifar10', 'cifar100'] :
                net = nn.Sequential()
                with net.name_scope():
                    net.add(nn.Dense(256, activation='relu'))
                    net.add(nn.Dense(128, activation='relu'))
                    net.add(nn.Dense(64, activation='relu'))
                    if self.regression:
                        net.add(nn.Dense(1))
                    else:
                        if dataset == 'cifar100':
                            net.add(nn.Dense(100))
                        else:
                            net.add(nn.Dense(10))
            else:
                raise NotImplementedError
        elif self.model == 'cnn':
            if dataset in ['mnist', 'fashionmnist']:
                if not self.regression:
                    net = NetMNIST()
                else:
                    raise NotImplementedError
            elif dataset == 'cifar10':
                if not self.regression:
                    #net = model_zoo.get_model('cifar_resnet20_v1', pretrained=False)
                    net = model_zoo.get_model('cifar_wideresnet16_10', pretrained=False)
                else:
                    raise NotImplementedError
            elif dataset == 'cifar100':
                if not self.regression:
                    net = get_cifar100_wide_resnet(28, width_factor=10, drop_rate=0.3)
                else:
                    raise NotImplementedError


        net.initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)

        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
        self.net = net
        self.trainer = trainer
        self.res = []

        if self.regression:
            self.softmax_cross_entropy_loss = gluon.loss.L2Loss()
        else:
            self.softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    def eval(self, train_data, loadckpt=None):
        net = self.net 
        ctx = self.ctx 

        if loadckpt:
            self.net.load_params(loadckpt, ctx=ctx)
        list_v = []

        for batch in train_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            for x, y in zip(data, label):
                z = net(x)
                loss = self.softmax_cross_entropy_loss(z, y)
                list_v += list(loss.asnumpy())
        return list_v

    def eval_acc(self, data, loadckpt=None):
        ctx = self.ctx 
        net = self.net 
        if loadckpt:
            net.load_params(loadckpt, ctx=ctx)
        if self.regression:
            metric = mx.metric.MSE()
        else:
            metric = mx.metric.Accuracy()
        for batch in data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            for x,y in zip(data, label):
                z = net(x)
                outputs.append(z)
            metric.update(label, outputs)
        name, acc = metric.get()
        metric.reset()
        return acc 

    def train(self, train_data, val_data, val_data_bad, savefilename=None, train_bad=None):
        epoch = self.epoch
        trainer = self.trainer
        net = self.net 
        ctx = self.ctx 
        batch_size = self.batch_size
        net.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx, force_reinit=True)

        earlystopping = -1
        for i in range(epoch):
            if i == int(epoch*0.6):
                trainer.set_learning_rate(trainer.learning_rate * 0.2)

            # Reset the train data iterator.
            #train_data.reset()
            # Loop over the train data iterator.
            cnt = 0
            for batch in train_data:
                # Splits train data into multiple slices along batch_axis
                # and copy each slice into a context.
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                # Splits train labels into multiple slices along batch_axis
                # and copy each slice into a context.
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                # Inside training scope
                with ag.record():
                    for x, y in zip(data, label):
                        z = net(x)
                        loss = self.softmax_cross_entropy_loss(z, y)
                        loss.backward()
                # Make one step of parameter update. Trainer needs to know the
                # batch size of data to normalize the gradient by 1/batch_size.
                trainer.step(batch[0].shape[0])
                print('iter:', cnt)
                cnt += 1
            # Gets the evaluation result.
            acc = self.eval_acc(train_data)
            acc_train_bad = self.eval_acc(train_bad) 
            logging.info('training acc at epoch %d: accuracy = %f'%(i, acc))
            logging.info('bad training sample in the original training set: acc = %f'%(acc_train_bad))
            
            acc_unknown = self.eval_acc(val_data)
            logging.info('testing set true: acc = %f'%(acc_unknown))

            acc_val = self.eval_acc(val_data_bad)
            self.res.append(acc_val)
            if self.regression:
                if len(self.res) <= 1 or self.res[-1] < min(self.res[:-1]):
                    self.net.save_params(savefilename)
                    logging.info('saved current best model at epoch %d'%(i))
                    earlystopping = i 
            else:
                if len(self.res) <= 1 or self.res[-1] > max(self.res[:-1]):
                    self.net.save_params(savefilename)
                    logging.info('saved current best model at epoch %d, acc: %f'%(i, self.res[-1]))
                    earlystopping = i
            logging.info('epoch %d, acc: %f, min: %f, max: %f'%(i, self.res[-1], min(self.res), max(self.res)))
            if i - earlystopping > 100:
                logging.info('early stopping at iteration: %d'%(i))
                break
        logging.info('acc: %f, min: %f, max: %f'%(self.res[-1], min(self.res), max(self.res)))
        return self.res 
