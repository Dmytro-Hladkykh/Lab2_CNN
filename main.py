import os
import torch
from config import Config
from data import get_data_loaders, get_class_labels
from model import CNN
from train import train_model, plot_training_curves
from utils import setup_logger
from PIL import Image
from torchvision import transforms

def main(config):
    # Setup logger
    logger = setup_logger()

    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(config, "data")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.train_new_model:
        logger.info("Training a new model...")
        model = CNN().to(device)
        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader,
                                                               config.num_epochs, config.lr, device, logger)
        plot_training_curves(train_losses, val_losses, val_accuracies)
        torch.save(model.state_dict(), config.model_path)
    else:
        if os.path.exists(config.model_path):
            logger.info("Using pre-trained model...")
            model = CNN().to(device)
            model.load_state_dict(torch.load(config.model_path))
        else:
            logger.error("No pre-trained model found. Set 'train_new_model' to True to train a new model.")
            return

    # Perform on example
    if os.path.exists("cat.jpg"):
        logger.info("Performing on example - cat.jpg...")
        image = Image.open("cat.jpg").convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        
        # Get the class labels
        class_labels = get_class_labels()
        predicted_label = class_labels[predicted_class.item()] 

        logger.info(f"Predicted class for the example image: {predicted_label}")
    else:
        logger.error("cat.jpg not found.")

if __name__ == "__main__":
    config = Config()
    main(config)