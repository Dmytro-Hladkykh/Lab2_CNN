import os
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def get_class_labels():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def combine_batches_physically(batch_dir, selected_batches):
    combined_dir = f"combined_batches_{'-'.join(map(str, selected_batches))}"
    if os.path.exists(combined_dir):
        logger.info(f"Using existing combined directory: {combined_dir}")
        return combined_dir

    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(os.path.join(combined_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(combined_dir, "test", "images"), exist_ok=True)

    test_labels_file = os.path.join(combined_dir, "test", "labels.txt")
    open(test_labels_file, 'w').close()  

    logger.info(f"Combining batches {', '.join(map(str, selected_batches))} into {combined_dir}")

    for batch_idx in selected_batches:
        batch_path = os.path.join(batch_dir, f"batch_{batch_idx}")
        for img_file in os.listdir(os.path.join(batch_path, "images")):
            src_path = os.path.join(batch_path, "images", img_file)
            dst_path = os.path.join(combined_dir, "images", img_file)
            shutil.copy(src_path, dst_path)

        with open(os.path.join(combined_dir, "labels.txt"), "a") as f:
            with open(os.path.join(batch_path, "labels.txt"), "r") as batch_labels:
                f.writelines(batch_labels.readlines())

    return combined_dir

def split_combined_dataset(combined_dir, train_val_split):
    dataset = CustomDataset(combined_dir)
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    test_dataset = CustomDataset(os.path.join(combined_dir, "test"))  

    return train_dataset, val_dataset, test_dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = sorted(os.listdir(os.path.join(data_dir, "images")))
        self.label_file = os.path.join(data_dir, "labels.txt")
        self.labels, self.classes = self.load_labels()  
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_labels(self):
        labels = {}
        classes = []  
        with open(self.label_file, "r") as f:
            for line in f:
                idx, label = line.strip().split(",")
                labels[idx] = label
                if label not in classes:
                    classes.append(label)
        return labels, classes  

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, "images", self.image_files[idx])
        image = Image.open(img_path)
        image = self.transform(image)
        label = self.labels[self.image_files[idx].split("_")[1].split(".")[0]]
        return image, label

def load_cifar10(root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10 = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    return cifar10

def split_dataset_into_batches(dataset, num_batches, batch_dir):
    if os.path.exists(batch_dir):
        logger.info(f"Using existing batch directory: {batch_dir}")
        return

    os.makedirs(batch_dir, exist_ok=True)

    logger.info(f"Splitting dataset into {num_batches} batches and saving to {batch_dir}")

    for i in range(num_batches):
        batch_path = os.path.join(batch_dir, f"batch_{i+1}")
        os.makedirs(batch_path, exist_ok=True)
        os.makedirs(os.path.join(batch_path, "images"), exist_ok=True)

    batch_size = len(dataset) // num_batches
    for i, (img, label) in enumerate(dataset):
        batch_idx = i // batch_size
        batch_path = os.path.join(batch_dir, f"batch_{batch_idx+1}")
        img_path = os.path.join(batch_path, "images", f"img_{i}.png")
        Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()).save(img_path)
        with open(os.path.join(batch_path, "labels.txt"), "a") as f:
            f.write(f"{i},{dataset.classes[label]}\n")

def get_data_loaders(config, root):
    cifar10 = load_cifar10(root)

    batch_dir = "batches"
    split_dataset_into_batches(cifar10, config.num_batches, batch_dir)

    combined_dir = combine_batches_physically(batch_dir, config.selected_batches)

    original_dataset = CustomDataset(combined_dir)
    train_dataset, val_dataset, test_dataset = split_combined_dataset(combined_dir, config.train_val_split)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, original_dataset))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, original_dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, original_dataset))

    return train_loader, val_loader, test_loader

def collate_fn(batch, dataset):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor([dataset.classes.index(label) for label in labels])
    return images, labels