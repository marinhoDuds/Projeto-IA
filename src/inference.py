import torch

from PIL import Image
from torchvision import transforms

from models.cnn_classification import AgeClassificationModel
from models.cnn_regression import AgeRegressionModel
from models.cnn_multi import AgeMultiModel

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
            pred = torch.argmax(output, dim=1).item()

        elif model_type == "r":
            pred = output.item()

        else:
            age_pred = output[0].item()
            class_pred = torch.argmax(output[1], dim=1).item()

            pred = {
                "age": age_pred,
                "class": class_pred
            }

    return pred


def inference(path, model_path, device, img_size):
    model = load_model(model_path, device)
    transform = transforms.Compose([
                                transforms.Resize((img_size, img_size)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])
                            ])

    pred = predict(model, path, transform, device)
    print(pred)