import os
import torch
from config import Config
from data import get_data_loaders, get_class_labels
from model import ResNet50
from train import train_model, plot_training_curves
from utils import setup_logger
from PIL import Image
from torchvision import transforms

def main(config):
    logger = setup_logger()

    train_loader, val_loader, test_loader = get_data_loaders(config, "data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.train_new_model:
        logger.info("Training a new model...")
        model = ResNet50().to(device)
        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader,
                                                               config.num_epochs, config.lr, device, logger)
        plot_training_curves(train_losses, val_losses, val_accuracies)
        torch.save(model.state_dict(), config.model_path)
    else:
        if os.path.exists(config.model_path):
            logger.info("Using pre-trained model...")
            model = ResNet50().to(device)
            model.load_state_dict(torch.load(config.model_path))
        else:
            logger.error("No pre-trained model found. Set 'train_new_model' to True to train a new model.")
            return

if __name__ == "__main__":
    config = Config()
    main(config)