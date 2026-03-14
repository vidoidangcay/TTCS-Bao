from dataset import get_dataloaders

train_loader, test_loader = get_dataloaders()

for images, labels in train_loader:
    print(images.shape)
    print(labels)
    break