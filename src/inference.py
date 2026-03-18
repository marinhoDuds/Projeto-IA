import torch

from PIL import Image
from torchvision import transforms

from models.cnn_classification import AgeClassificationModel
from models.cnn_regression import AgeRegressionModel
from models.cnn_multi import AgeMultiModel

CLASS_LABELS = {
    0: "criança pequena",
    1: "criança",
    2: "adolescente",
    3: "jovem adulto",
    4: "adulto",
    5: "idoso"
}

def load_model(model_path, device):
    checkpoint = torch.load(model_path)

    model_type = checkpoint["type"]
    if model_type == "c":
        model = AgeClassificationModel()

    elif model_type == "r":
        model = AgeRegressionModel()

    else:
        model = AgeMultiModel()

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model


def predict(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    model_type = model.type
    with torch.no_grad():
        output = model(image)

        if model_type == "c":
            class_idx = torch.argmax(output, dim=1).item()
            pred = CLASS_LABELS[class_idx]

        elif model_type == "r":
            pred = output.item()

        else:
            age_pred = output[0].item()
            class_idx = torch.argmax(output[1], dim=1).item()

            pred = {
                "age": age_pred,
                "class": CLASS_LABELS[class_idx]
            }

    return pred


def inference(path, model_path, device, img_size):
    model = load_model(model_path, device)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
    ])
    pred = predict(model, path, transform, device)

    return pred