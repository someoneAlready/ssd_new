#!/usr/bin/env python

import mxnet as mx
import argparse


def plot_train(network):

    from symbol.symbol_factory_mobi import get_symbol_train
    net = get_symbol_train(network, 300,
            num_classes=4, nms_thresh=0.5,
            force_suppres=True, nms_topk=400)

    mx.viz.plot_network(net, node_attrs={"hide_weights":"true",
            "fixedsize":'false', "shape":'oval'} ).render('train_')

def plot_test(network):
    from symbol.symbol_factory_mobi import get_symbol

    net = get_symbol(network, 300,
            num_classes=4, nms_thresh=0.5,
            force_suppres=True, nms_topk=400)
    data_shape = (1, 3, 300, 300)

    '''
    mx.viz.plot_network_detail(net, shape={"data":data_shape}, 
            node_attrs={"hide_weights":"true",
            "fixedsize":'false', "shape":'oval'} ).render(network)
    '''

    mx.viz.plot_network(net, shape={"data":data_shape}, 
            node_attrs={"hide_weights":"true",
            "fixedsize":'false', "shape":'oval'} ).render(network)


#plot_train('mobiTmp')
plot_test('mobiTmp')

