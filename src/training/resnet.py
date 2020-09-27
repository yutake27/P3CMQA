import chainer
import chainer.functions as F
import chainer.links as L
import chainermn.links as MNL

class Building(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, bn_kwargs={}):
        w = chainer.initializers.HeNormal()
        super(Building, self).__init__()

        if 'comm' in bn_kwargs:
            norm_layer = MNL.MultiNodeBatchNormalization
        else:
            norm_layer = L.BatchNormalization
        
        self.use_conv = (n_in != n_out)
        with self.init_scope():
            self.bn1 = norm_layer(n_in, **bn_kwargs)
            self.conv1 = L.Convolution3D(n_in, n_mid, 3, stride, 1, True, w)
            self.bn2 = norm_layer(n_mid, **bn_kwargs)
            self.conv2 = L.Convolution3D(n_mid, n_out, 3, 1, 1, True, w)
            if self.use_conv:
                self.conv3 = L.Convolution3D(n_in, n_out, 1, stride, 0, True, w)
            
            #self.conv1 = L.Convolution3D(n_in, n_mid, 3, stride, 1, True, w)
            #self.bn1 = norm_layer(n_mid, **bn_kwargs)
            #self.conv2 = L.Convolution3D(n_mid, n_out, 3, 1, 1, True, w)
            #self.bn2 = norm_layer(n_out, **bn_kwargs)
            # if use_conv:
                # self.conv3 = L.Convolution3D(n_in, n_out, 1, stride, 0, True, w)
                # self.bn3 = norm_layer(n_out, **bn_kwargs)


    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = h + self.conv3(x) if self.use_conv else h + x

        #h = F.relu(self.bn1(self.conv1(x)))
        #h = self.bn2(self.conv2(h))
        #h = h + self.bn3(self.conv3(x)) if self.use_conv else h + x
        return h


class BottleNeck(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, bn_kwargs={}):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()

        if 'comm' in bn_kwargs:
            norm_layer = MNL.MultiNodeBatchNormalization
        else:
            norm_layer = L.BatchNormalization

        self.use_conv = (n_in != n_out)
        with self.init_scope():
            self.bn1 = norm_layer(n_in, **bn_kwargs)
            self.conv1 = L.Convolution3D(n_in, n_mid, 1, stride, 0, True, w)
            self.bn2 = norm_layer(n_mid, **bn_kwargs)
            self.conv2 = L.Convolution3D(n_mid, n_mid, 3, 1, 1, True, w)
            self.bn3 = norm_layer(n_mid, **bn_kwargs)
            self.conv3 = L.Convolution3D(n_mid, n_out, 1, 1, 0, True, w)
            if self.use_conv:
                self.conv4 = L.Convolution3D(n_in, n_out, 1, stride, 0, True, w)

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))
        h = h + self.conv4(x) if self.use_conv else h + x
        return h


class Block(chainer.ChainList):

    def __init__(self, n_in, n_mid, n_out, n_bottlenecks, stride=2, block=BottleNeck, bn_kwargs={}):
        super(Block, self).__init__()
        self.add_link(block(n_in, n_mid, n_out, stride, bn_kwargs))
        for _ in range(n_bottlenecks - 1):
            self.add_link(block(n_out, n_mid, n_out, bn_kwargs=bn_kwargs))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class ResNet18(chainer.Chain):

    def __init__(self, n_class=1, n_blocks=[2, 2, 2, 2], bn_kwargs={}):
        super(ResNet18, self).__init__()
        w = chainer.initializers.HeNormal()

        if 'comm' in bn_kwargs:
            norm_layer = MNL.MultiNodeBatchNormalization
        else:
            norm_layer = L.BatchNormalization

        with self.init_scope():
            self.conv1 = L.Convolution3D(None, 64, 3, 1, 1, True, w)
            self.res3 = Block(64, 64, 64, n_blocks[0], 1, Building, bn_kwargs)
            self.res4 = Block(64, 128, 128, n_blocks[1], 2, Building, bn_kwargs)
            self.res5 = Block(128, 256, 256, n_blocks[2], 2, Building, bn_kwargs)
            self.res6 = Block(256, 512, 512, n_blocks[3], 2, Building, bn_kwargs)
            self.fc7 = L.Linear(None, n_class)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_3d(h, h.shape[2:])
        h = self.fc7(h)
        return h


class ResNet18_reduce(chainer.Chain):

    def __init__(self, n_class=1, n_blocks=[2, 2, 2, 2], bn_kwargs={}):
        super(ResNet18, self).__init__()
        w = chainer.initializers.HeNormal()

        if 'comm' in bn_kwargs:
            norm_layer = MNL.MultiNodeBatchNormalization
        else:
            norm_layer = L.BatchNormalization

        with self.init_scope():
            self.conv1 = L.Convolution3D(None, 64, 3, 1, 1, True, w)
            self.res3 = Block(64, 64, 64, n_blocks[0], 1, Building, bn_kwargs)
            self.res4 = Block(64, 128, 128, n_blocks[1], 2, Building, bn_kwargs)
            self.res5 = Block(128, 256, 256, n_blocks[2], 2, Building, bn_kwargs)
            self.res6 = Block(256, 512, 512, n_blocks[3], 2, Building, bn_kwargs)
            self.fc7 = L.Linear(None, n_class)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_3d(h, h.shape[2:])
        h = self.fc7(h)
        return h


class ResNet10(ResNet18):

    def __init__(self, n_class=1, bn_kwargs={}):
        super(ResNet10, self).__init__(n_class, [1,1,1,1], bn_kwargs)


class ResNet(chainer.Chain):

    def __init__(self, n_class=1, n_blocks=[3, 4, 6, 3], bn_kwargs={}):
        super(ResNet, self).__init__()
        w = chainer.initializers.HeNormal()

        if 'comm' in bn_kwargs:
            norm_layer = MNL.MultiNodeBatchNormalization
        else:
            norm_layer = L.BatchNormalization

        with self.init_scope():
            self.conv1 = L.Convolution3D(None, 64, 3, 1, 1, True, w)
            self.res3 = Block(64, 64, 256, n_blocks[0], 1, bn_kwargs=bn_kwargs)
            self.res4 = Block(256, 128, 512, n_blocks[1], 2, bn_kwargs=bn_kwargs)
            self.res5 = Block(512, 256, 1024, n_blocks[2], 2, bn_kwargs=bn_kwargs)
            self.res6 = Block(1024, 512, 2048, n_blocks[3], 2, bn_kwargs=bn_kwargs)
            self.fc7 = L.Linear(None, n_class)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_3d(h, h.shape[2:])
        h = self.fc7(h)
        return h


class ResNet14(ResNet):
    def __init__(self, n_class=10, bn_kwargs={}):
        super(ResNet14, self).__init__(n_class, [1, 1, 1, 1], bn_kwargs)


class ResNet26(ResNet):

    def __init__(self, n_class=10, bn_kwargs={}):
        super(ResNet26, self).__init__(n_class, [2, 2, 2, 2], bn_kwargs)


class ResNet50(ResNet):

    def __init__(self, n_class=10):
        super(ResNet50, self).__init__(n_class, [3, 4, 6, 3], bn_kwargs)


class ResNet101(ResNet):

    def __init__(self, n_class=10):
        super(ResNet101, self).__init__(n_class, [3, 4, 23, 3], bn_kwargs)


class ResNet152(ResNet):

    def __init__(self, n_class=10):
        super(ResNet152, self).__init__(n_class, [3, 8, 36, 3])


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1, 39, 32, 32, 32).astype(np.float32)
    model = ResNet18(10)
    y = model(x)
    print(y.shape)
    from chainer.functions import softmax_cross_entropy
    print(softmax_cross_entropy(y, np.array([3])))
