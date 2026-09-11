"""Portable native Torch execution of the retained Keras Functional checkpoint.

No Keras, TensorFlow, optree, or deserialized Python code is imported or executed.
Only the explicitly supported layer/config types below are interpreted. Kernels
keep their original H5 layout, allowing lossless shape-preserving checkpoint
updates. The public model interface is NHWC; convolution computation is NCHW
with channels-last storage, matching the recorded Keras Torch backend.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _histories(value):
    if isinstance(value, dict):
        if value.get('class_name') == '__keras_tensor__':
            return [tuple(value['config']['keras_history'])]
        return [item for nested in value.values() for item in _histories(nested)]
    if isinstance(value, list):
        return [item for nested in value for item in _histories(nested)]
    return []


def _refs(value):
    if len(value) == 3 and isinstance(value[0], str):
        return [tuple(value)]
    return [tuple(ref) for ref in value]


def _activation(x, name):
    if name == 'linear':
        return x
    if name == 'relu':
        return torch.relu(x)
    if name == 'sigmoid':
        return torch.sigmoid(x)
    raise ValueError(f'Unsupported activation: {name}')


class _Layer(nn.Module):
    def __init__(self, spec, group):
        super().__init__()
        self.kind, self.config = spec['class_name'], spec['config']
        self.keras_name = self.config['name']
        self.variable_paths = []
        self.weights = nn.ParameterList()
        variables = group['vars']
        for key in sorted(variables, key=int):
            array = np.asarray(variables[key])
            if array.dtype != np.float32:
                raise ValueError(f'Unsupported checkpoint dtype: {variables[key].name} {array.dtype}')
            self.variable_paths.append(variables[key].name)
            self.weights.append(nn.Parameter(torch.from_numpy(array.copy()),
                                            requires_grad=bool(self.config.get('trainable', True))))
        if self.kind == 'BatchNormalization' and self.config.get('trainable', True):
            raise ValueError('Portable port requires the original frozen BatchNormalization configuration')
        supported = {'InputLayer', 'CameraNormalization', 'Rescaling', 'Conv2D', 'DepthwiseConv2D',
                     'SeparableConv2D', 'BatchNormalization', 'ReLU', 'ZeroPadding2D', 'Add',
                     'UpSampling2D', 'Activation', 'Concatenate'}
        if self.kind not in supported:
            raise ValueError(f'Unsupported layer class: {self.kind}')
        if self.config.get('data_format', 'channels_last') != 'channels_last':
            raise ValueError('Only channels_last Keras layer configs are supported')

    def _conv(self, x, raw_kernel, *, depthwise=False, pointwise=False):
        config = self.config
        kernel = raw_kernel
        if depthwise:
            kernel = kernel.reshape(*kernel.shape[:2], 1, kernel.shape[2] * kernel.shape[3])
        kernel = kernel.permute(3, 2, 0, 1).contiguous(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
        strides = (1, 1) if pointwise else tuple(config.get('strides', (1, 1)))
        dilation = tuple(config.get('dilation_rate', (1, 1)))
        padding = 0
        if not pointwise and config['padding'] == 'same':
            pairs = []
            for size, window, stride, dilate in zip(x.shape[2:], kernel.shape[2:], strides, dilation):
                total = max(((size + stride - 1) // stride - 1) * stride + dilate * (window - 1) + 1 - size, 0)
                pairs.append((total // 2, total - total // 2))
            if all(left == right for left, right in pairs):
                padding = tuple(left for left, _ in pairs)
            else:
                x = F.pad(x, tuple(value for pair in reversed(pairs) for value in pair))
        elif not pointwise and config['padding'] != 'valid':
            raise ValueError(f'Unsupported convolution padding: {config["padding"]}')
        groups = x.shape[1] // kernel.shape[1]
        if x.shape[1] % kernel.shape[1]:
            raise ValueError('Kernel input channels do not divide input channels')
        return F.conv2d(x, kernel, stride=strides, padding=padding, dilation=dilation, groups=groups)

    def forward(self, inputs):
        config, weights, kind = self.config, self.weights, self.kind
        if kind == 'Add':
            result = inputs[0]
            for tensor in inputs[1:]:
                result = result + tensor
            return result
        if kind == 'Concatenate':
            if config['axis'] not in (-1, 3):
                raise ValueError('Only channel concatenation is supported')
            return torch.cat(inputs, dim=1)
        x = inputs[0]
        if kind == 'InputLayer':
            return x
        if kind == 'CameraNormalization':
            # Preserve the original sequential NHWC reduction order exactly.
            rgb = x.permute(0, 2, 3, 1)
            shifted = rgb - rgb[:, :1, :1, :]
            centered = shifted - shifted.mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
            variance = centered.square().mean(dim=2, keepdim=True).mean(dim=1, keepdim=True)
            std = variance.clamp_min(config['std_floor'] ** 2).sqrt()
            gray = (centered / std).mean(dim=-1, keepdim=True)
            return (.5 + config['target_std'] * gray).clamp(0, 1).repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
        if kind == 'Rescaling':
            return x * config['scale'] + config['offset']
        if kind in ('Conv2D', 'DepthwiseConv2D', 'SeparableConv2D'):
            x = self._conv(x, weights[0], depthwise=kind != 'Conv2D')
            if kind == 'SeparableConv2D':
                x = self._conv(x, weights[1], pointwise=True)
            if config['use_bias']:
                x = x + weights[-1].reshape(1, -1, 1, 1)
            return _activation(x, config['activation'])
        if kind == 'BatchNormalization':
            if config['axis'] not in (-1, 3):
                raise ValueError('Only channel BatchNormalization is supported')
            index = 0
            scale = weights[index] if config['scale'] else 1.
            index += int(config['scale'])
            offset = weights[index] if config['center'] else 0.
            index += int(config['center'])
            mean, variance = weights[index], weights[index + 1]
            multiplier = (variance + config['epsilon']).rsqrt() * scale
            bias = offset.reshape(1, -1, 1, 1) if isinstance(offset, torch.Tensor) else offset
            return (x - mean.reshape(1, -1, 1, 1)) * multiplier.reshape(1, -1, 1, 1) + bias
        if kind == 'ReLU':
            threshold, slope = config['threshold'], config['negative_slope']
            negative = -slope * torch.relu(-x + threshold) if slope else 0.
            result = torch.where(x > threshold, x, 0.) if threshold else torch.relu(x)
            if config['max_value'] is not None:
                result = result.clamp_max(config['max_value'])
            return result + negative
        if kind == 'ZeroPadding2D':
            (top, bottom), (left, right) = config['padding']
            return F.pad(x, (left, right, top, bottom))
        if kind == 'UpSampling2D':
            if config['interpolation'] != 'nearest':
                raise ValueError('Only nearest UpSampling2D is supported')
            return x.repeat_interleave(config['size'][0], dim=2).repeat_interleave(config['size'][1], dim=3)
        if kind == 'Activation':
            return _activation(x, config['activation'])
        raise AssertionError(kind)


class _Graph(nn.Module):
    def __init__(self, config, group):
        super().__init__()
        self.keras_name = config['name']
        self.inputs, self.outputs = _refs(config['input_layers']), _refs(config['output_layers'])
        self.sequence = []
        self.operations = nn.ModuleDict()
        by_name = {}
        for key, layer_group in group['layers'].items():
            name = layer_group['vars'].attrs['name']
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            if name in by_name:
                raise ValueError(f'Duplicate H5 layer name: {name}')
            by_name[name] = layer_group
        for index, spec in enumerate(config['layers']):
            name, kind = spec['config']['name'], spec['class_name']
            if name not in by_name:
                raise ValueError(f'Checkpoint variables not found for layer {name}')
            if kind == 'Functional':
                operation = _Graph(spec['config'], by_name[name])
            else:
                operation = _Layer(spec, by_name[name])
            key = str(index)
            self.operations[key] = operation
            if kind == 'InputLayer':
                continue
            nodes = spec['inbound_nodes']
            if len(nodes) != 1:
                raise ValueError('Reused or unconnected layers require an explicit port')
            refs = _histories(nodes[0]['args'])
            if any(node != 0 for _, node, _ in refs):
                raise ValueError('Only node-zero Functional references are supported')
            self.sequence.append((name, key, refs))

    @staticmethod
    def _get(values, ref):
        name, node, output = ref
        if node != 0:
            raise ValueError('Only node-zero outputs are supported')
        value = values[name]
        if isinstance(value, (list, tuple)):
            return value[output]
        if output != 0:
            raise ValueError('Requested nonexistent layer output')
        return value

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if len(inputs) != len(self.inputs):
            raise ValueError('Functional input count differs')
        values = {ref[0]: value for ref, value in zip(self.inputs, inputs)}
        for name, key, refs in self.sequence:
            values[name] = self.operations[key]([self._get(values, ref) for ref in refs])
        outputs = [self._get(values, ref) for ref in self.outputs]
        return outputs[0] if len(outputs) == 1 else outputs


class PortableWordModel(nn.Module):
    def __init__(self, config, weights, checkpoint_sha256):
        super().__init__()
        if config['class_name'] != 'Functional':
            raise ValueError('Checkpoint must contain a Functional model')
        self.checkpoint_sha256 = checkpoint_sha256
        self.architecture = config['config']
        self.graph = _Graph(config['config'], weights)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[-1] != 3:
            raise ValueError('Expected NHWC RGB input')
        output = self.graph(inputs.permute(0, 3, 1, 2))
        if isinstance(output, list):
            return [value.permute(0, 2, 3, 1) for value in output]
        return output.permute(0, 2, 3, 1)

    def freeze_backbone(self, frozen=True):
        found = False
        for module in self.graph.modules():
            if isinstance(module, _Graph) and module.keras_name == 'mobilenetv2_half':
                found = True
                for layer in module.modules():
                    if isinstance(layer, _Layer):
                        for parameter in layer.weights:
                            parameter.requires_grad_(not frozen and layer.config.get('trainable', True))
        if not found:
            raise ValueError('mobilenetv2_half backbone not found')
        return self


def load_keras_checkpoint(path):
    """Load retained data/config without importing Keras or evaluating code."""
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read('config.json'))
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as weights:
            return PortableWordModel(config, weights, _digest(path))


def parse_checkpoint(path):
    """Return full JSON config and layer-name-tuple -> ordered NumPy arrays.

    Nested keys include each enclosing model name, excluding the root model:
    ('mobilenetv2_half', 'Conv1') and ('fpn_4',), for example. Weight-free
    layers have empty lists. This helper does not instantiate Torch operations.
    """
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read('config.json'))
        arrays = {}
        with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as weights:
            def visit(group, prefix=()):
                for layer in group['layers'].values():
                    name = layer['vars'].attrs['name']
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    key = prefix + (name,)
                    if key in arrays:
                        raise ValueError(f'Duplicate layer identity: {key}')
                    arrays[key] = [np.asarray(layer['vars'][index]).copy() for index in sorted(layer['vars'], key=int)]
                    if 'layers' in layer:
                        visit(layer, key)
            visit(weights)
        return config, arrays


def save_keras_checkpoint(model, parent, path):
    """Write updated weights, keeping architecture and clearing stale optimizer.

    The parent must be the exact checkpoint used to construct this model. This
    is an uncompiled inference/continuation checkpoint; native Torch optimizer
    state should be saved separately by the trainer for exact training resume.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    if _digest(parent) != model.checkpoint_sha256:
        raise ValueError('Save parent differs from the checkpoint used to load this model')
    with zipfile.ZipFile(parent) as archive:
        config = json.loads(archive.read('config.json'))
        if config['config'] != model.architecture:
            raise ValueError('Checkpoint architecture changed')
        memory = io.BytesIO(archive.read('model.weights.h5'))
        with h5py.File(memory, 'r+') as weights:
            for module in model.modules():
                if isinstance(module, _Layer):
                    for variable_path, parameter in zip(module.variable_paths, module.weights):
                        array = parameter.detach().cpu().numpy()
                        if weights[variable_path].shape != array.shape or not np.isfinite(array).all():
                            raise ValueError(f'Invalid updated weight: {variable_path}')
                        weights[variable_path][...] = array
            if 'optimizer' in weights:
                del weights['optimizer']
        config['compile_config'] = {}
        with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as result:
            for item in archive.infolist():
                content = archive.read(item.filename)
                if item.filename == 'model.weights.h5':
                    content = memory.getvalue()
                elif item.filename == 'config.json':
                    content = json.dumps(config, separators=(',', ':')).encode('utf-8')
                result.writestr(item, content)


def joint_word_loss(truth, prediction, relative_sizes=True):
    """Native Torch equivalent of word_model.joint_word_loss, NHWC tensors."""
    heatmap = truth[..., :1]
    probability = prediction[..., :1].clamp(1e-5, 1 - 1e-5)
    centers = (heatmap >= .999).to(torch.float32)
    valid = (heatmap >= 0).to(torch.float32)
    normalizer = centers.sum().clamp_min(1.)
    positive = -centers * (1 - probability).square() * probability.log()
    negative = -(1 - centers) * valid * (1 - heatmap.clamp_min(0)).pow(4) * probability.square() * (1 - probability).log()
    localization = (positive + negative).sum() / normalizer
    if relative_sizes:
        offset_delta = truth[..., 1:3] - prediction[..., 1:3]
        size_delta = truth[..., 3:5].clamp_min(1e-3).log() - prediction[..., 3:5].clamp_min(1e-3).log()
        delta = torch.cat((offset_delta, size_delta), dim=-1)
    else:
        delta = (truth[..., 1:5] - prediction[..., 1:5]) * prediction.new_tensor([1., 1., 4., 2.])
    absolute = delta.abs()
    huber = torch.where(absolute < 1., .5 * delta.square(), absolute - .5)
    geometry = (huber * centers).sum() / normalizer
    bits = prediction[..., 5:].clamp(1e-5, 1 - 1e-5)
    known_bits = (truth[..., 5:] >= 0).to(torch.float32)
    labels = truth[..., 5:].clamp_min(0.)
    bit_bce = -(labels * bits.log() + (1 - labels) * (1 - bits).log())
    dot_normalizer = (centers * known_bits).sum().clamp_min(1.)
    dot_loss = (bit_bce * centers * known_bits).sum() / dot_normalizer
    return localization + 5 * geometry + 3 * dot_loss


word_relative_loss = joint_word_loss
