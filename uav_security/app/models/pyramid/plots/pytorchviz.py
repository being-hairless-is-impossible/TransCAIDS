import torch

from app.models.pyramid.model import PyramidTransformerKAN
from torchviz import make_dot


data = torch.randn(56)
cutoff = 10
import torch
from app.models.pyramid.model import PyramidTransformerKAN
from torchviz import make_dot

# Initialize model
model = PyramidTransformerKAN(input_dim_cyber=128, input_dim_physical=128, num_classes=10)

# Prepare input data
x_cyber = torch.randn(128, 64)
x_physical = torch.randn(128, 64)
y = model(x_cyber, x_physical)

# Create graph with only key parameters (filtering out too many details)
dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=False, show_saved=False, max_attr_chars = 10)

# Render the graph
dot.render("model_architecture", format="png")
