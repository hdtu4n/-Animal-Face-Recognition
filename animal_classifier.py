import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Định nghĩa lại class Net giống như trong notebook
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # ===== CNN LAYERS =====
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # Conv layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # Conv layer 2  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Conv layer 3

        self.pooling = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.relu = nn.ReLU()              # Activation function
        
        # ===== FULLY CONNECTED LAYERS =====
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128 * 16 * 16), 128)  # Hidden layer
        self.output = nn.Linear(128, 3)                 # Output layer (3 classes)

    def forward(self, x):
        # ===== CNN FEATURE EXTRACTION =====
        x = self.conv1(x)    # Conv1: (3,128,128) → (32,128,128)
        x = self.pooling(x)  # Pool:  (32,128,128) → (32,64,64)
        x = self.relu(x)     # ReLU activation
        
        x = self.conv2(x)    # Conv2: (32,64,64) → (64,64,64)
        x = self.pooling(x)  # Pool:  (64,64,64) → (64,32,32)
        x = self.relu(x)     # ReLU activation
        
        x = self.conv3(x)    # Conv3: (64,32,32) → (128,32,32)
        x = self.pooling(x)  # Pool:  (128,32,32) → (128,16,16)
        x = self.relu(x)     # ReLU activation
        
        # ===== CLASSIFICATION HEAD =====
        x = self.flatten(x)  # Flatten: (128,16,16) → (32768,)
        x = self.linear(x)   # Dense: (32768,) → (128,)
        x = self.output(x)   # Output: (128,) → (3,)
        return x


class AnimalClassifier:
    def __init__(self, model_path="model/animal_faces_classification"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Khởi tạo label encoder (giống notebook)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["cat", "dog", "wild"])

        # Transform giống notebook
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

        # Load model
        self.model = Net().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_image(self, image_path):
        """Predict một ảnh và trả về class + confidence"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            class_name = self.label_encoder.inverse_transform([predicted.item()])[0]
            confidence_score = confidence.item()

        return class_name, confidence_score

    def predict_with_all_scores(self, image_path):
        """Predict và trả về tất cả scores cho 3 classes"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        results = {}
        for i, class_name in enumerate(["cat", "dog", "wild"]):
            results[class_name] = probabilities[i]

        return results


# Test function
if __name__ == "__main__":
    classifier = AnimalClassifier()

    # Test với ảnh của bạn
    result, confidence = classifier.predict_image("D:/NCKH/Demo/123.jpg")
    print(f"Prediction: {result}, Confidence: {confidence:.2%}")

    # Test với all scores
    all_scores = classifier.predict_with_all_scores("D:/NCKH/Demo/123.jpg")
    print("All scores:", all_scores)
