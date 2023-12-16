from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models


class AnomalyDetectionModel(nn.Module):
    def __init__(self):
        super(AnomalyDetectionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 3)

    def forward(self, x):
        return self.resnet(x)


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model_path = "model.pth"
model = AnomalyDetectionModel()
model.load_state_dict(torch.load(model_path))

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


class_labels = ["Normal", "Anomalous"]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            input_image = process_image(filepath)

            with torch.no_grad():
                output = model(input_image)

            prediction = torch.argmax(output).item()
            print(prediction)
            result = class_labels[prediction]

            return render_template(
                "index.html",
                result=result,
                image_path=filename,
            )

    return render_template("index.html", result=None, image_path=None)


if __name__ == "__main__":
    app.run(debug=True)
