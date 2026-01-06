# blink_model.py
import torch
from torchvision import transforms
from blinklinmult.models import DenseNet121

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_blink_model(weight_path):
    model = DenseNet121(output_dim=1, weights=None).to(DEVICE)
    print(">>> Loading weight from:", weight_path)
    state = torch.load(r"D:\UIT\Xử lý ảnh\Predict-eye-state-streamlit\model_weight\densenet121-union-64.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def predict_eye_state(model, eye_img):
    x = preprocess(eye_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()
    return "CLOSED" if p > 0.5 else "OPEN", p
