import chainer
import chainer.functions as F
import chainer.links as L
from octconv import OctConv
from octconv import OctConv_BN
from octconv import OctConv_BN_ACT
from octconv import oct_add, oct_function


class Building(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, alpha=0.25, bn_kwargs={}):
        w = chainer.initializers.HeNormal()
        super(Building, self).__init__()

        self.use_conv = (n_in != n_out)
        with self.init_scope():
            self.conv_bn_act1 = OctConv_BN_ACT(n_in, n_mid, 3, stride, 1, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)
            self.conv_bn2 = OctConv_BN(n_mid, n_out, 3, 1, 1, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)
            if self.use_conv:
                self.conv_bn3 = OctConv_BN(n_in, n_out, 1, stride, 0, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)
        
    
    def __call__(self, x):
        h = self.conv_bn_act1(x)
        h = self.conv_bn2(h)
        h = oct_add(h, self.conv_bn3(x)) if self.use_conv else oct_add(h, x)
        h = oct_function(F.relu)(h)
        return h

class BottleNeck(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, alpha=0.25, bn_kwargs={}):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()

        self.use_conv = (n_in != n_out)
        with self.init_scope():
            self.conv_bn_act1 = OctConv_BN_ACT(n_in, n_mid, 1, stride, 0, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)
            self.conv_bn_act2 = OctConv_BN_ACT(n_mid, n_mid, 3, 1, 1, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)
            self.conv_bn3 = OctConv_BN(n_mid, n_out, 1, 1, 0, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)
            if self.use_conv:
                self.conv_bn4 = Conv_BN(
                    n_in, n_out, 1, stride, 0, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)

    def __call__(self, x):
        h = self.conv_bn_act1(x)
        h = self.conv_bn_act2(h)
        h = self.conv_bn3(h)
        h = oct_add(h, self.conv_bn4(x)) if self.use_conv else oct_add(h, x)
        h = oct_function(F.relu)(h)
        return h

class Block(chainer.ChainList):

    def __init__(self, n_in, n_mid, n_out, n_bottlenecks, stride=2, alpha=0.25, block=BottleNeck, bn_kwargs={}):
        super(Block, self).__init__()
        self.add_link(block(n_in, n_mid, n_out, stride, alpha=alpha, bn_kwargs=bn_kwargs))
        for _ in range(n_bottlenecks - 1):
            self.add_link(block(n_out, n_mid, n_out, alpha=alpha, bn_kwargs=bn_kwargs))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class OctResNet18(chainer.Chain):

    def __init__(self, n_class=1, n_blocks=[2, 2, 2, 2], alpha=0.25, bn_kwargs={}):
        super(OctResNet18, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv_bn_act1 = OctConv_BN_ACT(None, 64, 3, 1, 1, nobias=True, initialW=w, alpha_out=alpha, bn_kwargs=bn_kwargs)
            self.res3 = Block(64, 64, 64, n_blocks[0], 1, alpha=alpha, block=Building, bn_kwargs=bn_kwargs)
            self.res4 = Block(64, 128, 128, n_blocks[1], 2, alpha=alpha, block=Building, bn_kwargs=bn_kwargs)
            self.res5 = Block(128, 256, 256, n_blocks[2], 2, alpha=alpha, block=Building, bn_kwargs=bn_kwargs)
            self.res6 = Block(256, 512, 512, n_blocks[3], 2, alpha=0, block=Building, bn_kwargs=bn_kwargs)
            self.fc7 = L.Linear(None, n_class)

    def __call__(self, x_h, x_l=None):
        h = self.conv_bn_act1((x_h, x_l))
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_3d(h, h.shape[2:])
        h = self.fc7(h)
        return h


class OctResNet(chainer.Chain):

    def __init__(self, n_class=10, n_blocks=[3, 4, 6, 3], alpha=0.25, bn_kwargs={}):
        super(OctResNet, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv_bn_act1 = OctConv_BN_ACT(None, 64, 3, 1, 1, nobias=True, initialW=w, alpha_out=alpha) #TODO why origin pad==0??
            self.res3 = Block(64, 64, 256, n_blocks[0], 1, alpha=alpha, bn_kwargs=bn_kwargs)
            self.res4 = Block(256, 128, 512, n_blocks[1], 2, alpha=alpha, bn_kwargs=bn_kwargs)
            self.res5 = Block(512, 256, 1024, n_blocks[2], 2, alpha=alpha, bn_kwargs=bn_kwargs)
            self.res6 = Block(1024, 512, 2048, n_blocks[3], 2, alpha=0, bn_kwargs=bn_kwargs)
            self.fc7 = L.Linear(None, n_class)

    def __call__(self, x_h, x_l=None):
        h = self.conv_bn_act1((x_h, x_l))
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_3d(h, h.shape[2:])
        h = self.fc7(h)
        return h


class OctResNet50(OctResNet):

    def __init__(self, n_class=10):
        super(OctResNet50, self).__init__(n_class, [3, 4, 6, 3])


class OctResNet101(OctResNet):

    def __init__(self, n_class=10):
        super(OctResNet101, self).__init__(n_class, [3, 4, 23, 3])


class OctResNet152(OctResNet):

    def __init__(self, n_class=10):
        super(OctResNet152, self).__init__(n_class, [3, 8, 36, 3])


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 14, 32, 32, 32).astype(np.float32)
    y = np.random.randn(1, 24, 16, 16, 16).astype(np.float32)
    model = OctResNet18(1)
    y = model(x,y)
    print(y.shape)
