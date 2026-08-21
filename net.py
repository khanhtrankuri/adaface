from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Conv2d, Linear
from torch.nn import BatchNorm1d, BatchNorm2d
from torch.nn import ReLU, Sigmoid
from torch.nn import Module
from torch.nn import PReLU


def build_model(model_name='ir_50'):
    """Build an AdaFace backbone.

    Names ending in ``_dla`` use Conv-BN residual blocks.  They are intended for
    a newly trained (or fine-tuned) TensorRT/DLA model and deliberately do not
    share the exact graph of the published, pre-activation checkpoints.
    """
    builders = {
        'ir_18': IR_18,
        'ir_34': IR_34,
        'ir_50': IR_50,
        'ir_101': IR_101,
        'ir_se_50': IR_SE_50,
        'ir_18_dla': IR_18_DLA,
        'ir_34_dla': IR_34_DLA,
        'ir_50_dla': IR_50_DLA,
        'ir_101_dla': IR_101_DLA,
        'ir_se_50_dla': IR_SE_50_DLA,
    }
    try:
        builder = builders[model_name]
    except KeyError:
        raise ValueError('not a correct model name: {}'.format(model_name))
    return builder(input_size=(112, 112))

def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


def convert_legacy_state_dict_for_dla(state_dict):
    """Remap a legacy backbone state dict to initialize a DLA backbone.

    The pre-convolution BN in every residual branch and the pre-linear BN in
    the output head have no equivalent in the new graph, so those tensors are
    discarded.  The returned weights are only an initialization and require
    fine-tuning plus fresh INT8 calibration.
    """
    converted = {}
    for key, value in state_dict.items():
        new_key = key

        if '.res_layer.' in key:
            prefix, suffix = key.split('.res_layer.', 1)
            module_index, separator, parameter_name = suffix.partition('.')
            if module_index.isdigit():
                module_index = int(module_index)
                if module_index == 0:
                    continue
                new_key = '{}.res_layer.{}'.format(prefix, module_index - 1)
                if separator:
                    new_key += '.' + parameter_name

        if 'output_layer.' in new_key:
            prefix, suffix = new_key.split('output_layer.', 1)
            module_index, separator, parameter_name = suffix.partition('.')
            if module_index.isdigit():
                module_index = int(module_index)
                if module_index == 0:
                    continue
                # Dropout and Flatten have no tensors. Only Linear (3 -> 2)
                # and the final BatchNorm1d (4 -> 3) reach this branch.
                if module_index >= 3:
                    module_index -= 1
                new_key = '{}output_layer.{}'.format(prefix, module_index)
                if separator:
                    new_key += '.' + parameter_name

        converted[new_key] = value

    return converted


class Flatten(Module):
    """ Flat tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class LinearBlock(Module):
    """ Convolution block without no-linear activation layer
    """
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Module):
    """ Global Norm-Aware Pooling block
    """
    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Module):
    """ Global Depthwise Convolution block
    """
    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(in_c, in_c,
                                     groups=in_c,
                                     kernel=(7, 7),
                                     stride=(1, 1),
                                     padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(in_c, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size, affine=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SEModule(Module):
    """ SE block
    """
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x



class BasicBlockIR(Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRDLA(Module):
    """Post-normalization IR block for TensorRT/DLA deployment.

    Unlike the published IR block, every normalization in the residual branch
    follows a convolution.  TensorRT can therefore fold Conv-BN pairs during
    engine construction and calibrate them as a single weighted operation.
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRDLA, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class BottleneckIR(Module):
    """ BasicBlock with bottleneck for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIRDLA(Module):
    """Post-normalization bottleneck used by the large DLA backbones."""
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRDLA, self).__init__()
        reduction_channel = depth // 4
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            Conv2d(in_channel, reduction_channel, (1, 1), (1, 1), 0, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, reduction_channel, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(reduction_channel),
            PReLU(reduction_channel),
            Conv2d(reduction_channel, depth, (1, 1), stride, 0, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BasicBlockIRSEDLA(BasicBlockIRDLA):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSEDLA, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class BottleneckIRSEDLA(BottleneckIRDLA):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSEDLA, self).__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", SEModule(depth, 16))


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] +\
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2),
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=8),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(in_channel=64, depth=256, num_units=3),
            get_block(in_channel=256, depth=512, num_units=24),
            get_block(in_channel=512, depth=1024, num_units=36),
            get_block(in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir', block_layout='legacy'):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or ir_se
            block_layout: ``legacy`` keeps the published BN-Conv graph;
                ``dla`` uses foldable Conv-BN pairs.
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        assert block_layout in ['legacy', 'dla'], \
            "block_layout should be legacy or dla"
        self.block_layout = block_layout
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir' and block_layout == 'legacy':
                unit_module = BasicBlockIR
            elif mode == 'ir_se' and block_layout == 'legacy':
                unit_module = BasicBlockIRSE
            elif mode == 'ir':
                unit_module = BasicBlockIRDLA
            else:
                unit_module = BasicBlockIRSEDLA
            output_channel = 512
        else:
            if mode == 'ir' and block_layout == 'legacy':
                unit_module = BottleneckIR
            elif mode == 'ir_se' and block_layout == 'legacy':
                unit_module = BottleneckIRSE
            elif mode == 'ir':
                unit_module = BottleneckIRDLA
            else:
                unit_module = BottleneckIRSEDLA
            output_channel = 2048

        output_modules = []
        if block_layout == 'legacy':
            output_modules.append(BatchNorm2d(output_channel))
        output_modules.extend([Dropout(0.4), Flatten()])
        if input_size[0] == 112:
            output_modules.append(Linear(output_channel * 7 * 7, 512))
        else:
            output_modules.append(Linear(output_channel * 14 * 14, 512))
        output_modules.append(BatchNorm1d(512, affine=False))
        self.output_layer = Sequential(*output_modules)

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        initialize_weights(self.modules())


    def forward_embedding(self, x):
        """Return the unnormalized embedding.

        Export this method through :class:`DLAEmbeddingModel` and perform L2
        normalization/cosine similarity in FP32 outside DLA.  A sum reduction is
        otherwise likely to be quantized or moved to a GPU fallback subgraph.
        """
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)

    def forward(self, x):
        x = self.forward_embedding(x)
        if self.block_layout == 'legacy':
            norm = torch.norm(x, 2, 1, True)
            return torch.div(x, norm), norm

        # FP32 accumulation prevents overflow/underflow in FP16. Keep this out of
        # an INT8 DLA engine by exporting DLAEmbeddingModel instead.
        fp32_embedding = x.float()
        norm = torch.norm(fp32_embedding, 2, 1, True).clamp_min(1e-12)
        output = torch.div(fp32_embedding, norm)
        return output, norm


class DLAEmbeddingModel(Module):
    """Export wrapper that leaves L2 normalization and cosine outside DLA."""
    def __init__(self, backbone):
        super(DLAEmbeddingModel, self).__init__()
        if backbone.block_layout != 'dla':
            raise ValueError('DLAEmbeddingModel requires a *_dla backbone')
        self.backbone = backbone

    def forward(self, x):
        return self.backbone.forward_embedding(x)



def IR_18(input_size):
    """ Constructs a ir-18 model.
    """
    model = Backbone(input_size, 18, 'ir')

    return model


def IR_34(input_size):
    """ Constructs a ir-34 model.
    """
    model = Backbone(input_size, 34, 'ir')

    return model


def IR_50(input_size):
    """ Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """ Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """ Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_200(input_size):
    """ Constructs a ir-200 model.
    """
    model = Backbone(input_size, 200, 'ir')

    return model


def IR_SE_50(input_size):
    """ Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """ Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """ Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model


def IR_SE_200(input_size):
    """ Constructs a ir_se-200 model.
    """
    model = Backbone(input_size, 200, 'ir_se')

    return model


def IR_18_DLA(input_size):
    return Backbone(input_size, 18, 'ir', block_layout='dla')


def IR_34_DLA(input_size):
    return Backbone(input_size, 34, 'ir', block_layout='dla')


def IR_50_DLA(input_size):
    return Backbone(input_size, 50, 'ir', block_layout='dla')


def IR_101_DLA(input_size):
    return Backbone(input_size, 100, 'ir', block_layout='dla')


def IR_SE_50_DLA(input_size):
    return Backbone(input_size, 50, 'ir_se', block_layout='dla')

