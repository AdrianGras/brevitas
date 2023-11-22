# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import nn

from brevitas import config
from brevitas.core.scaling.standalone import ConstScaling
from brevitas.core.scaling.standalone import ParameterScaling
from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.equalize import EqualizeGraph
from brevitas.graph.fixed_point import CollapseConsecutiveConcats
from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.graph.fixed_point import MoveSplitBatchNormBeforeCat
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph.quantize_impl import add_output_quant_handler
from brevitas.graph.quantize_impl import inp_placeholder_handler
from brevitas.graph.quantize_impl import layer_handler
from brevitas.graph.quantize_impl import residual_handler
from brevitas.graph.standardize import DisableLastReturnQuantTensor
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph.standardize import RemoveStochasticModules
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.nn import quant_layer
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloatMaxInit
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat



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

def align_input_quant(
        module, shared_quant_identity, shared_quant_identity_name, quant_identity_map, align_sign):
    """
    Based on the input module, the function decides how to align its output.
    """
    # If it is a QuantIdentity already, simply modify tensor_quant or the scaling implementations
    # based on whether we need to align the sign or not
    if isinstance(module, qnn.QuantIdentity):
        if align_sign or module.is_quant_act_signed == shared_quant_identity.is_quant_act_signed:
            return shared_quant_identity
        else:
            assert not module.is_quant_act_signed and shared_quant_identity.is_quant_act_signed
            quant_module_class, quant_module_kwargs = quant_identity_map['unsigned']
            return (
                quant_module_class,
                {
                    **quant_module_kwargs,
                    'scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .scaling_impl,
                    'int_scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .int_scaling_impl})
    elif hasattr(module, 'output_quant'):
        return (type(module), {'output_quant': shared_quant_identity})
    # If it is a QuantAct where the scaling can be determined through stats (thus through calibration),
    # then adapt its act_quant according to align_sign.
    elif hasattr(module, 'act_quant') and not isinstance(
            module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl,
        (ParameterScaling, ConstScaling)):
        module_type = type(module)
        if align_sign:
            partial_config = {
                'signed':
                    shared_quant_identity.act_quant.is_signed,
                'tensor_quant':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant}
        else:
            partial_config = {
                'scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .scaling_impl,
                'int_scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .int_scaling_impl}
        injector = module.act_quant.quant_injector.let(**partial_config)
        return module_type(act_quant=injector, return_quant_tensor=True)
    # In all other cases, return the name of the QuantIdentity that will be added at the output of
    # the module
    else:
        return shared_quant_identity_name


def preprocess_for_quantize(
        model,
        trace_model=True,
        relu6_to_relu=True,
        equalize_iters=0,
        equalize_merge_bias=True,
        merge_bn=True,
        equalize_bias_shrinkage: str = 'vaiq',
        equalize_scale_computation: str = 'maxabs'):

    training_state = model.training
    model.eval()

    if trace_model:
        model = symbolic_trace(model)
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    if relu6_to_relu:
        model = ModuleToModuleByClass(nn.ReLU6, nn.ReLU).apply(model)
    model = MeanMethodToAdaptiveAvgPool2d().apply(model)
    model = CollapseConsecutiveConcats().apply(model)
    model = MoveSplitBatchNormBeforeCat().apply(model)
    if merge_bn:
        model = MergeBatchNorm().apply(model)
    model = RemoveStochasticModules().apply(model)
    model = EqualizeGraph(
        iterations=equalize_iters,
        merge_bias=equalize_merge_bias,
        bias_shrinkage=equalize_bias_shrinkage,
        scale_computation_type=equalize_scale_computation).apply(model)
    model.train(training_state)
    return model

def quantize_model(
        model,
        backend,
        act_bit_width,
        weight_bit_width,
        bias_bit_width,
        scaling_per_output_channel,
        act_quant_percentile,
        act_quant_type,
        scale_factor_type,
        weight_bit_width_map,
        activation_bit_width_map,
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



def layerwise_quantize(graph_model, compute_layer_map=LAYERWISE_COMPUTE_LAYER_MAP):
    ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
    config.IGNORE_MISSING_KEYS = True
    training_state = graph_model.training
    graph_model.eval()
    graph_model = layer_handler(graph_model, layer_map=compute_layer_map, requantize_output=False)
    graph_model.train(training_state)
    config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
    return graph_model
