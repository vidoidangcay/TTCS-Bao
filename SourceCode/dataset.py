import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import config

PRETRAINED_MODELS = {"resnet18"}


def get_transforms(model_name="baseline", train=False):
    image_size = config.IMAGE_SIZE
    if model_name in PRETRAINED_MODELS:
        normalize = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
        channel_transform = transforms.Grayscale(num_output_channels=3)
    else:
        normalize = transforms.Normalize(config.GRAYSCALE_MEAN, config.GRAYSCALE_STD)
        channel_transform = transforms.Grayscale()

    transform_list = []
    if train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0))
        ])
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.extend([
        channel_transform,
        transforms.ToTensor(),
        normalize
    ])
    return transforms.Compose(transform_list)


def get_dataloaders(model_name="baseline"):
    train_transform = get_transforms(model_name, train=True)
    val_transform = get_transforms(model_name, train=False)
    test_transform = get_transforms(model_name, train=False)

    full_dataset = datasets.ImageFolder(root=config.TRAIN_DIR)
    total_samples = len(full_dataset)
    val_size = int(total_samples * config.VAL_SPLIT)
    train_size = total_samples - val_size

    generator = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(total_samples, generator=generator).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    train_dataset = Subset(
        datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform),
        train_indices
    )
    val_dataset = Subset(
        datasets.ImageFolder(root=config.TRAIN_DIR, transform=val_transform),
        val_indices
    )
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    for images, labels in train_loader:
        print(images.shape)
        print(labels)
        break