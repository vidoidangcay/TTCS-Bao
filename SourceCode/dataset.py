import config
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def get_dataloaders():

    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, test_loader