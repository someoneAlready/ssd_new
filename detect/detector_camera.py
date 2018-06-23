from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
import cv2
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol,  label_names = None,  context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
        self.mod.set_params(args, auxs, allow_missing=True)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, im):
        im = cv2.resize(im, (self.data_shape, self.data_shape), 
                interpolation = cv2.INTER_CUBIC)
        im = im[:,:,::-1]

        im = im.transpose(2, 0, 1).astype(np.float32) 
        im = im - np.reshape(self.mean_pixels, (3, 1, 1))
        data = im[np.newaxis,...]

        self.mod.forward( Batch( [mx.nd.array(data)] ))
        output = self.mod.get_outputs()[0].asnumpy()

        det = output[0, :, :]
        res = det[np.where( det[:, 0] >= 0)[0]]

        return res

