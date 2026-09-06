"""Read actual saved backbone padding/stride without importing an ML backend."""
import hashlib
import json
import math
from pathlib import Path
import zipfile

work = Path('.braille-training-work/word-scan-v1-20260905')
checkpoint = work / 'selected-v13/best-word.keras'
with zipfile.ZipFile(checkpoint) as archive:
    configuration_bytes = archive.read('config.json')
configuration = json.loads(configuration_bytes)
backbone = next(layer for layer in configuration['config']['layers'] if layer['name'] == 'mobilenetv2_half')
states = {}
details = []


def tensors(value):
    if isinstance(value, dict):
        if value.get('class_name') == '__keras_tensor__':
            yield value['config']
        else:
            for item in value.values():
                yield from tensors(item)
    elif isinstance(value, list):
        for item in value:
            yield from tensors(item)


for layer in backbone['config']['layers']:
    kind, name, config = layer['class_name'], layer['name'], layer['config']
    if kind == 'InputLayer':
        state = {'size': config['batch_shape'][1:3], 'jump': [1, 1], 'origin': [0., 0.], 'conv_support': [1, 1]}
    else:
        inputs = list(tensors(layer['inbound_nodes']))
        parents = [states[item['keras_history'][0]] for item in inputs]
        for item, parent in zip(inputs, parents):
            assert item['shape'][1:3] == parent['size'], (name, item['shape'], parent)
        state = {key: list(value) for key, value in parents[0].items()}
        if kind in ('Conv2D', 'DepthwiseConv2D'):
            for axis in range(2):
                size, jump = state['size'][axis], state['jump'][axis]
                kernel = (config['kernel_size'][axis]-1)*config['dilation_rate'][axis]+1
                stride = config['strides'][axis]
                if config['padding'] == 'same':
                    output = math.ceil(size/stride)
                    before = max((output-1)*stride+kernel-size, 0)//2
                else:
                    assert config['padding'] == 'valid'
                    output, before = (size-kernel)//stride+1, 0
                state['size'][axis] = output
                state['origin'][axis] += ((kernel-1)/2-before)*jump
                state['conv_support'][axis] += (kernel-1)*jump
                state['jump'][axis] *= stride
        elif kind == 'ZeroPadding2D':
            for axis in range(2):
                before, after = config['padding'][axis]
                state['size'][axis] += before+after
                state['origin'][axis] -= before*state['jump'][axis]
        elif kind == 'Add':
            assert all(all(parent[key] == state[key] for key in ('size', 'jump', 'origin')) for parent in parents)
            state['conv_support'] = [max(parent['conv_support'][axis] for parent in parents) for axis in range(2)]
        else:
            assert kind in ('BatchNormalization', 'ReLU'), (kind, name)
        details.append({'name': name, 'class': kind, **state})
    states[name] = state

outputs = {item[0]: states[item[0]] for item in backbone['config']['output_layers']}
assert outputs['block_3_expand_relu']['origin'] == [3., 3.]
assert outputs['block_3_expand_relu']['jump'] == [4, 4]
assert outputs['block_3_expand_relu']['size'] == [48, 120]
report = {
    'status': 'passed',
    'scope': 'Nominal convolution sampling lattices from actual saved backbone graph. No model inference, training, image sampling or TFLite proof.',
    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'saved_config_sha256': hashlib.sha256(configuration_bytes).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_coordinate_space': 'Pixel-index centers; first input sample is index0.',
    'feature_axes': ['y', 'x'],
    'backbone_outputs': outputs,
    'stride4_input_pixel_to_feature_index': '(input_pixel_index - 3) / 4',
    'limitations': [
        'Coordinates of annotation box edges and resized occupied sample centers are not interchangeable; the caller must define the tap coordinate convention.',
        'FPN adds nearest-upsampled coarser maps with piecewise repeated centers. Origin3 is the nominal fine-branch lattice, not a claim of a single physical receptive center for the fused feature.',
        'Per-image normalization and coarse features have wider dependencies; geometric convolution support is not a guarantee of local dot evidence.',
        'A future sampler still needs analytic ramp/boundary and nonzero converted-graph parity tests.',
    ],
    'layers': details,
}
destination = work / 'feature-lattice-audit.json'
destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'passed', 'outputs': outputs, 'report_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))
