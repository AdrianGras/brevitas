import pytest

import numpy as np
import torch
import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.general import RemoveStaticGraphInputs
from finn.transformation.double_to_single_float import DoubleToSingleFloat

from brevitas_examples.imagenet_classification import quant_mobilenet_v1_4b
from brevitas.onnx import export_finn_onnx


INPUT_SIZE = (1, 3, 224, 224)
ATOL = 1e-3
SEED = 0

@pytest.mark.parametrize("pretrained", [True])
def test_mobilenet_v1_4b(pretrained):
    finn_onnx = "mobilenet_v1_4b.onnx"
    mobilenet = quant_mobilenet_v1_4b(pretrained)
    mobilenet.eval()
    #load a random test vector
    np.random.seed(SEED)
    numpy_tensor = np.random.random(size=INPUT_SIZE).astype(np.float32)
    # run using PyTorch/Brevitas
    torch_tensor = torch.from_numpy(numpy_tensor).float()
    # do forward pass in PyTorch/Brevitas
    expected = mobilenet(torch_tensor).detach().numpy()
    export_finn_onnx(mobilenet, INPUT_SIZE, finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    # run using FINN-based execution
    inp_name = model.graph.input[0].name
    input_dict = {inp_name: numpy_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    assert np.isclose(produced, expected, atol=ATOL).all()
