# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import torch

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import COMPUTE_LAYER_MAP
from brevitas.graph.quantize import LAYERWISE_COMPUTE_LAYER_MAP
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import QUANT_ACT_MAP
from brevitas.graph.quantize import QUANT_IDENTITY_MAP
from brevitas.graph.quantize import quantize
from brevitas.graph.target.flexml import FLEXML_COMPUTE_LAYER_MAP
from brevitas.graph.target.flexml import FLEXML_QUANT_ACT_MAP
from brevitas.graph.target.flexml import FLEXML_QUANT_IDENTITY_MAP
from brevitas.graph.target.flexml import quantize_flexml
from brevitas.inject.enum import RestrictValueType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFixedPoint
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
import brevitas.nn as qnn
import torch.nn as nn
from brevitas.inject import value
from brevitas.quant import Int8WeightPerTensorFixedPoint
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Uint8ActPerTensorFloatMaxInit
import tqdm

def get_layer_map(weight_bit_width_map, activation_bit_width_map, weight_bit_width, act_bit_width, backend):
    class VariableBitWidthInt8WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
        @value
        def bit_width(module):
            if module in weight_bit_width_map:
                return weight_bit_width_map[module]
            else:
                weight_bit_width_map[module] = weight_bit_width
                return weight_bit_width
    WEIGHT_QUANT = VariableBitWidthInt8WeightPerTensorFixedPoint

    class VariableBitWidthUint8ActPerTensorFloat(Uint8ActPerTensorFloat):
        @value
        def bit_width(module):
            if module in activation_bit_width_map:
                return activation_bit_width_map[module]
            else:
                activation_bit_width_map[module] = act_bit_width
                return act_bit_width
    
    U_ACT_QUANT = VariableBitWidthUint8ActPerTensorFloat

    class VariableBitWidthInt8ActPerTensorFloat(Int8ActPerTensorFloat):
        @value
        def bit_width(module):
            if module in activation_bit_width_map:
                return activation_bit_width_map[module]
            else:
                activation_bit_width_map[module] = act_bit_width
                return act_bit_width
    
    ACT_QUANT = VariableBitWidthInt8ActPerTensorFloat

    class VariableBitWidthInt8ActPerTensorFloatMinMaxInit(Int8ActPerTensorFloatMinMaxInit):
        @value
        def bit_width(module):
            if module in activation_bit_width_map:
                return activation_bit_width_map[module]
            else:
                activation_bit_width_map[module] = act_bit_width
                return act_bit_width
    
    ACT_QUANT_MIN_MAX = VariableBitWidthInt8ActPerTensorFloatMinMaxInit

    class VariableBitWidthUint8ActPerTensorFloatMaxInit(Uint8ActPerTensorFloatMaxInit):
        @value
        def bit_width(module):
            if module in activation_bit_width_map:
                return activation_bit_width_map[module]
            else:
                activation_bit_width_map[module] = act_bit_width
                return act_bit_width
    
    U_ACT_QUANT_MIN_MAX = VariableBitWidthUint8ActPerTensorFloatMaxInit
    
    COMPUTE_LAYER_MAP = {
    nn.AvgPool2d:
        None,
    nn.MultiheadAttention: (
        qnn.QuantMultiheadAttention,
        {
            'in_proj_weight_quant': WEIGHT_QUANT,
            'in_proj_bias_quant': Int32Bias,
            'attn_output_weights_quant': U_ACT_QUANT,
            'q_scaled_quant': ACT_QUANT,
            'k_transposed_quant': ACT_QUANT,
            'v_quant': ACT_QUANT,
            'out_proj_input_quant': ACT_QUANT,
            'out_proj_weight_quant': WEIGHT_QUANT,
            'out_proj_bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.RNN: (
        qnn.QuantRNN,
        {
            'weight_quant': WEIGHT_QUANT,
            'bias_quant': Int32Bias,
            'io_quant': ACT_QUANT,
            'gate_acc_quant': ACT_QUANT,
            'return_quant_tensor': True}),
    nn.LSTM: (
        qnn.QuantLSTM,
        {
            'weight_quant': WEIGHT_QUANT,
            'bias_quant': Int32Bias,
            'io_quant': ACT_QUANT,
            'gate_acc_quant': ACT_QUANT,
            'sigmoid_quant': U_ACT_QUANT,
            'tanh_quant': ACT_QUANT,
            'cell_state_quant': ACT_QUANT,
            'return_quant_tensor': True}),
    nn.Conv1d: (
        qnn.QuantConv1d,
        {
            'weight_quant': WEIGHT_QUANT,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': WEIGHT_QUANT,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose1d: (
        qnn.QuantConvTranspose1d,
        {
            'weight_quant': WEIGHT_QUANT,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'weight_quant': WEIGHT_QUANT,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.Linear: (
        qnn.QuantLinear,
        {
            'weight_quant': WEIGHT_QUANT,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True})}

    LAYERWISE_COMPUTE_LAYER_MAP = {
        nn.AvgPool2d:
            None,
        nn.MultiheadAttention: (
            qnn.QuantMultiheadAttention,
            {
                'in_proj_input_quant': ACT_QUANT,
                'in_proj_weight_quant': WEIGHT_QUANT,
                'in_proj_bias_quant': Int32Bias,
                'attn_output_weights_quant': U_ACT_QUANT,
                'q_scaled_quant': ACT_QUANT,
                'k_transposed_quant': ACT_QUANT,
                'v_quant': ACT_QUANT,
                'out_proj_input_quant': ACT_QUANT,
                'out_proj_weight_quant': WEIGHT_QUANT,
                'out_proj_bias_quant': Int32Bias,
                'return_quant_tensor': False}),
        nn.LSTM: (
            qnn.QuantLSTM,
            {
                'weight_quant': WEIGHT_QUANT,
                'bias_quant': Int32Bias,
                'io_quant': ACT_QUANT,
                'gate_acc_quant': ACT_QUANT,
                'sigmoid_quant': U_ACT_QUANT,
                'tanh_quant': ACT_QUANT,
                'cell_state_quant': ACT_QUANT,
                'return_quant_tensor': False}),
        nn.RNN: (
            qnn.QuantRNN,
            {
                'weight_quant': WEIGHT_QUANT,
                'bias_quant': Int32Bias,
                'io_quant': ACT_QUANT,
                'gate_acc_quant': ACT_QUANT,
                'return_quant_tensor': False}),
        nn.Conv1d: (
            qnn.QuantConv1d,
            {
                'input_quant': ACT_QUANT,
                'weight_quant': WEIGHT_QUANT,
                'bias_quant': Int32Bias,
                'return_quant_tensor': False}),
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                'input_quant': ACT_QUANT,
                'weight_quant': WEIGHT_QUANT,
                'bias_quant': Int32Bias,
                'return_quant_tensor': False}),
        nn.ConvTranspose1d: (
            qnn.QuantConvTranspose1d,
            {
                'input_quant': ACT_QUANT,
                'weight_quant': WEIGHT_QUANT,
                'bias_quant': Int32Bias,
                'return_quant_tensor': False}),
        nn.ConvTranspose2d: (
            qnn.QuantConvTranspose2d,
            {
                'input_quant': ACT_QUANT,
                'weight_quant': WEIGHT_QUANT,
                'bias_quant': Int32Bias,
                'return_quant_tensor': False}),
        nn.Linear: (
            qnn.QuantLinear,
            {
                'input_quant': ACT_QUANT,
                'weight_quant': WEIGHT_QUANT,
                'bias_quant': Int32Bias,
                'return_quant_tensor': False})}

    UNSIGNED_ACT_TUPLE = (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid)

    QUANT_ACT_MAP = {
        nn.ReLU: (qnn.QuantReLU, {
            'act_quant': U_ACT_QUANT, 'return_quant_tensor': True}),
        nn.ReLU6: (
            qnn.QuantReLU, {
                'act_quant': U_ACT_QUANT_MIN_MAX, 'max_val': 6.,
                'return_quant_tensor': True}),
        nn.Hardtanh: (
            qnn.QuantHardTanh,
            {
                'act_quant': ACT_QUANT_MIN_MAX,
                'max_val': lambda module: module.max_val,
                'min_val': lambda module: module.min_val,
                'return_quant_tensor': True}),
        nn.Sigmoid:
            (qnn.QuantSigmoid, {
                'act_quant': U_ACT_QUANT,
                'return_quant_tensor': True,}),}

    QUANT_IDENTITY_MAP = {
        'signed':
            (qnn.QuantIdentity, {
                'act_quant': ACT_QUANT, 'return_quant_tensor': True}),
        'unsigned':
            (qnn.QuantIdentity, {
                'act_quant': U_ACT_QUANT, 'return_quant_tensor': True}),}
    LAYER_MAP = {
    'generic': [COMPUTE_LAYER_MAP, QUANT_ACT_MAP, QUANT_IDENTITY_MAP],
    'layerwise': [LAYERWISE_COMPUTE_LAYER_MAP],
    'flexml': [FLEXML_COMPUTE_LAYER_MAP, FLEXML_QUANT_ACT_MAP, FLEXML_QUANT_IDENTITY_MAP]}

    return LAYER_MAP[backend]

QUANTIZE_MAP = {'layerwise': layerwise_quantize, 'generic': quantize, 'flexml': quantize_flexml}

ASYMMETRIC_ACT_QUANT_MAP = {
        'generic': ShiftedUint8ActPerTensorFloat,
        'layerwise': ShiftedUint8ActPerTensorFloat,
        'flexml': ShiftedUint8ActPerTensorFixedPoint}

BIAS_BIT_WIDTH_MAP = {'int32': Int32Bias, 'int16': Int16Bias}


def quantize_model(
        model,
        backend,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        act_quant_type,
        scale_factor_type,
        weight_bit_width_map,
        activation_bit_width_map,
        act_bit_width = {},
        weight_bit_width = {},
        weight_narrow_range=False):
    
    # Define what quantize function to use and, based on the given configuration, its arguments
    quantize_fn = QUANTIZE_MAP[backend]

    act_quant_asym = None
    if act_quant_type == 'asymmetric':
        act_quant_asym = ASYMMETRIC_ACT_QUANT_MAP[backend]

    maps = update_quant_maps(
        get_layer_map(weight_bit_width_map, activation_bit_width_map,weight_bit_width,act_bit_width,backend),
        scale_factor_type=scale_factor_type,
        bias_bit_width=bias_bit_width,
        scaling_per_output_channel=scaling_per_output_channel,
        act_quant_percentile=act_quant_percentile,
        act_quant_asym=act_quant_asym,
        act_bit_width=act_bit_width,
        weight_bit_width=weight_bit_width,
        weight_narrow_range=weight_narrow_range)

    if len(maps) == 3:
        # Generic and flexml requires three mappings for quantization
        quantize_kwargs = {
            'compute_layer_map': maps[0], 'quant_act_map': maps[1], 'quant_identity_map': maps[2]}
    elif len(maps) == 1:
        # Layerwise requires only the compute layer mapping
        quantize_kwargs = {'compute_layer_map': maps[0]}

    quant_model = quantize_fn(model, **quantize_kwargs)
    return quant_model


def update_quant_maps(
        maps,
        scale_factor_type,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        act_quant_asym,
        act_bit_width,
        weight_bit_width,
        weight_narrow_range):
    """
    Starting from pre-defined quantizers, modify them to match the desired configuration
    """

    act_kwargs = {'high_percentile_q': act_quant_percentile}

    if act_quant_asym is not None:
        act_kwargs['act_quant'] = act_quant_asym
        act_kwargs['low_percentile_q'] = 100.0 - act_quant_percentile

    weight_kwargs = {
        'scaling_per_output_channel': scaling_per_output_channel,
        'narrow_range': weight_narrow_range}

    scale_factor_dict = {}
    if scale_factor_type == 'po2':
        scale_factor_dict['restrict_scaling_type'] = RestrictValueType.POWER_OF_TWO
        scale_factor_dict['restrict_value_float_to_int_impl'] = CeilSte
    elif scale_factor_type == 'float32':
        scale_factor_dict['restrict_scaling_type'] = RestrictValueType.FP

    act_kwargs.update(scale_factor_dict)
    weight_kwargs.update(scale_factor_dict)

    def weight_kwargs_prefix(prefix):
        return {prefix + k: v for k, v in weight_kwargs.items()}

    def act_kwargs_prefix(prefix):
        updated_kwargs = {}
        for k, v in act_kwargs.items():
            key = k
            if prefix != '':
                key = prefix + key.replace('act_', '')
            updated_kwargs[key] = v
        return updated_kwargs

    bias_quant = BIAS_BIT_WIDTH_MAP[bias_bit_width]
    for map in maps:
        for k, v in map.items():
            if v is None:
                # Non quantized layer, continue
                continue
            if issubclass(v[0], QuantWBIOL):
                map[k][1].update(weight_kwargs_prefix('weight_'))
                map[k][1]['bias_quant'] = bias_quant
                if act_quant_asym is not None:
                    map[k][1]['return_quant_tensor'] = False
                if 'input_quant' in v[1].keys():
                    # Add kwargs arguments to input_quant, if present
                    map[k][1].update(act_kwargs_prefix('input_'))
            elif v[0] == QuantMultiheadAttention:
                map[k][1].update(weight_kwargs_prefix('in_proj_'))
                map[k][1].update(weight_kwargs_prefix('out_proj_'))
                map[k][1].update(act_kwargs_prefix('attn_output_weights_'))
                map[k][1].update(act_kwargs_prefix('q_scaled_'))
                map[k][1].update(act_kwargs_prefix('k_transposed_'))
                map[k][1].update(act_kwargs_prefix('v_'))
                map[k][1].update(act_kwargs_prefix('out_proj_input_'))
                map[k][1]['in_proj_bias_quant'] = bias_quant
                map[k][1]['out_proj_bias_quant'] = bias_quant
                if act_quant_asym is not None:
                    map[k][1]['return_quant_tensor'] = False
                if 'in_proj_input_quant' in v[1].keys():
                    # Add kwargs arguments to input_quant, if present
                    map[k][1].update(act_kwargs_prefix('in_proj_input_'))
            elif 'act_quant' in v[1].keys():
                # Add kwargs argument to activation quantizers.
                v[1].update(act_kwargs_prefix(''))

    return maps


def calibrate(calib_loader, model, bias_corr=True):
    """
    Perform calibration and bias correction, if enabled
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with calibration_mode(model):
            for i, (images, target) in tqdm(enumerate(calib_loader), total=len(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)

        if bias_corr:
            with bias_correction_mode(model):
                for i, (images, target) in tqdm(enumerate(calib_loader), total=len(calib_loader)):
                    images = images.to(device)
                    images = images.to(dtype)
                    model(images)
