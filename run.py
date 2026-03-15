import torch
import argparse

from src.train import train
from src.inference import inference

from models.cnn_classification import AgeClassificationModel
from models.cnn_regression import AgeRegressionModel
from models.cnn_multi import AgeMultiModel

IMG_SIZE = 128
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
PATIENCE = 5
CLASSIFICATION = "none"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Age estimation project")
subparsers = parser.add_subparsers(dest="command", required=True)

train_parser = subparsers.add_parser("train", help="Train the model")
train_parser.add_argument("-p", "--path", type=str, required=True, help="Path to dataset")
train_parser.add_argument("-m", "--model", type=str, required=True, help="Type of the model to train")

infer_parser = subparsers.add_parser("inference", help="Run inference")
infer_parser.add_argument("-p", "--path", type=str, required=True, help="Path to an image or a set of images")
infer_parser.add_argument("-m", "--model", type=str, required=True, help="Path to a trained model")

args = parser.parse_args()

if args.command == "train":
    if args.model == "c":
        model = AgeClassificationModel()
    elif args.model == "r":
        model = AgeRegressionModel()
    else:
        model = AgeMultiModel()

    train(model, args.path, device, IMG_SIZE, BATCH_SIZE, EPOCHS, LR, PATIENCE, CLASSIFICATION)

elif args.command == "inference":
    inference(args.path, args.model, device, IMG_SIZE)