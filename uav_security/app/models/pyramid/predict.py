import numpy as np
import torch
from app.models.pyramid.model import PyramidTransformerKAN, PyramidTransformer, PyramidConvLSTM

def model_predict(inputs: np.ndarray,
                  cutoff: int,
                  model: torch.nn.Module,
                  device = 'cuda' if torch.cuda.is_available() else 'cpu') -> np.ndarray:
    """
    Model prediction function for SHAP.

    :param inputs: Input data as NumPy array.
    :return: Model output probabilities as NumPy array.
    """
    device = torch.device(device)
    # Split the inputs back into cyber and physical
    cyber = torch.tensor(inputs[:cutoff], dtype=torch.float32).unsqueeze(1).to(device)
    physical = torch.tensor(inputs[cutoff:], dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(cyber, physical)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

    return probabilities


if __name__ == '__main__':
    data = np.array(
        [28122.89474, 28123.02875, 261.0, 98.0, 0.0, 320.0, 3.0, 1.0, 3.0, 1.0, 0.0, 0.0, 215.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0,
         0.002691, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 80, 65, 19309.0, 0, 0, 0, 0
         ])
    cutoff = 56
    # weight = '/home/shengguang/PycharmProjects/uav_security/app/outputs/fuse_pyramid_conv_transformer_kan_WARMUP/models/best_pyramid_model.pth'
    model = PyramidTransformerKAN(input_dim_cyber=cutoff,
                                  input_dim_physical=len(data)-cutoff,
                                  num_layers=5,
                                  attention_heads=4,
                                  num_classes=4)

