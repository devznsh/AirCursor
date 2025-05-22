# convert_to_openvino_fixed.py
import torch
import torch.nn as nn
from openvino import convert_model, save_model  # New OVC API
import os

# 1. Create model directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# 2. Define a proper dummy model
class HandLandmarker(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 63, kernel_size=3)  # 21 landmarks * 3 (x,y,z)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return self.conv2(x).view(-1, 21, 3)  # Shape: [batch, 21, 3]

# 3. Export to ONNX
print("Exporting to ONNX...")
model = HandLandmarker().eval()
dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    model,
    dummy_input,
    "models/hand_landmarker.onnx",
    input_names=["input"],
    output_names=["landmarks"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "landmarks": {0: "batch_size"}
    },
    opset_version=13
)

# 4. Convert to OpenVINO IR (using new OVC API)
print("Converting to OpenVINO IR...")
try:
    ov_model = convert_model("models/hand_landmarker.onnx")
    save_model(ov_model, "models/hand_landmarker.xml")
    print("✅ Conversion successful!")
    print("Generated files in models/ folder:")
    print(" - hand_landmarker.xml")
    print(" - hand_landmarker.bin")  # Created automatically
except Exception as e:
    print(f"❌ Conversion failed: {str(e)}")
    print("Try these fixes:")
    print("1. Update OpenVINO: pip install --upgrade openvino")
    print("2. Check write permissions in models/ folder")