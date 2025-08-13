import torch

model = Net()  # your model class
model.load_state_dict(torch.load('model_2048.pt'))
model.eval()

dummy_input = torch.randn(1, 20)  # batch size 1, 20 features (your board input size)
torch.onnx.export(model, dummy_input, "model_2048_for_js.onnx", input_names=['input'], output_names=['output'])
