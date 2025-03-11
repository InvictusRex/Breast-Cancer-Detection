import torch
import numpy as np

def convert_onnx_to_pytorch(onnx_session, pytorch_model, pth_model_path):
    dummy_input = torch.randn(1, 3, 224, 224)  # PyTorch input format (NCHW)
    
    # Convert input to ONNX format (NHWC)
    onnx_input = dummy_input.permute(0, 2, 3, 1).numpy()  # (1, 224, 224, 3)

    # Run ONNX inference with correct input shape
    ort_inputs = {onnx_session.get_inputs()[0].name: onnx_input}
    ort_outs = onnx_session.run(None, ort_inputs)

    # Convert PyTorch model and save
    torch.save(pytorch_model.state_dict(), pth_model_path)
    print(f"PyTorch model saved at: {pth_model_path}")
