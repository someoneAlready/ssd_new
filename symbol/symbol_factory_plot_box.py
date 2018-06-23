"""Presets for various network configurations"""
import logging
from symbol import symbol_builder_plot_box
import numpy as np

def get_scales(min_scale=0.2, max_scale=0.9,num_layers=6):

    ret = []
    for i in np.linspace(min_scale, max_scale, num_layers):
        ret.append([i])
    print(ret)
    return ret

def get_config(network, data_shape, **kwargs):
    """Configuration factory for various networks

    Parameters
    ----------
    network : str
        base network name, such as vgg_reduced, inceptionv3, resnet...
    data_shape : int
        input data dimension
    kwargs : dict
        extra arguments
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256, 256]
            strides = [-1, -1, 2, 2, 2, 2, 1]
            pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 512 else [x / 512.0 for x in
                [8, 16, 32, 64, 128, 256, 512]]
        else:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256]
            strides = [-1, -1, 2, 2, 1, 1]
            pads = [-1, -1, 1, 1, 0, 0]
            sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
        if not (data_shape == 300 or data_shape == 512):
            logging.warn('data_shape %d was not tested, use with caucious.' % data_shape)
        return locals()
    elif network == 'inceptionv3':
        if data_shape >= 448:
            from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', '']
            num_filters = [-1, -1, 512, 256, 256, 128]
            strides = [-1, -1, 2, 2, 2, 2]
            pads = [-1, -1, 1, 1, 1, 1]
            sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = -1
            steps = []
        else:
            from_layers = ['ch_concat_mixed_2_chconcat', 'ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '']
            num_filters = [-1, -1, -1, 256, 256, 128]
            strides = [-1, -1, -1, 2, 2, 2]
            pads = [-1, -1, -1, 1, 1, 1]
            sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                [1,2,.5], [1,2,.5]]
            normalizations = -1
            steps = []
        return locals()
    elif network == 'resnet50':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'mobilenet':
        from_layers = ['conv_12_relu', 'conv_14_relu', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 256]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.1, max_scale=0.6, num_layers=len(from_layers))
        ratios = [[1]] * 6
        normalizations = -1
        steps = []
        return locals()
    elif network == 'mobiTmp':
        from_layers = ['conv_12_relu', 'conv_14_relu', '']
        num_filters = [-1, -1, 512]
        strides = [-1, -1, 2]
        pads = [-1, -1, 1]
        sizes = get_scales(min_scale=0.05, max_scale=0.95, num_layers=len(from_layers))
        ratios = [[1]] * 3
        normalizations = -1
        steps = []
        return locals()
    elif network == 'pvanet':
        from_layers = ['_plus6', '_plus10', 'bsrlast_relu']
        num_filters = [-1, -1, -1]
        strides = [-1, -1, -1]
        pads = [-1, -1, -1]
        sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'mobilenet8':
        from_layers = ['conv_6_relu', 'conv_12_relu', 'conv_14_relu', '', '', '', '', '']
        num_filters = [-1, -1, -1, 512, 256, 256, 256, 256]
        strides = [-1, -1, -1, 2, 2, 2, 2, 2]
        pads = [-1, -1, -1, 1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
                  [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'densenet121':
        network = 'densenet'
        data_type = 'imagenet'
        units = [6, 12, 24, 16]
        num_stage = 4
        growth_rate = 32
        bottle_neck = True
        from_layers = ['DBstage3_concat24', 'DBstage4_concat16', '', '', '', '']
        num_filters = [-1, -1, 256, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'densenet-tiny':
        network = 'densenet'
        data_type = 'imagenet'
        units = [6, 12, 18, 12]
        num_stage = 4
        growth_rate = 16
        bottle_neck = True
        from_layers = ['DBstage2_concat12', 'DBstage3_concat18', '', '', '', '']
        num_filters = [-1, -1, 256, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(from_layers))
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    else:
        msg = 'No configuration found for %s with data_shape %d' % (network, data_shape)
        raise NotImplementedError(msg)

def get_symbol_train(network, data_shape, **kwargs):
    """Wrapper for get symbol for train

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol_train for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol_train(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder_plot_box.get_symbol_train(**config)

def get_symbol(network, data_shape, **kwargs):
    """Wrapper for get symbol for test

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol for more details
    """
    if network.startswith('legacy'):
        logging.warn('Using legacy model.')
        return symbol_builder.import_module(network).get_symbol(**kwargs)
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return symbol_builder_plot_box.get_symbol(**config)
