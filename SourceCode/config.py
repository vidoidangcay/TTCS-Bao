TRAIN_DIR = "archive/train"
TEST_DIR = "archive/test"

IMAGE_SIZE = 48
BATCH_SIZE = 32
NUM_WORKERS = 2
VAL_SPLIT = 0.2
NUM_CLASSES = 7  # angry, disgust, fear, happy, neutral, sad, surprise
LR = 1e-3
USE_REGULARIZATION = True  # Set False to disable all regularization
REGULARIZATION_TYPE = "l2"  # Options: "none", "l1", "l2"
REGULARIZATION_FACTOR = 5e-4  # Regularization strength for L1 or L2
USE_DROPOUT = True  # Set False to disable dropout layers in models
DROPOUT_PROB = 0.3  # Dropout probability used when USE_DROPOUT is True
EPOCHS = 50
DEVICE = "cuda"  

USE_WEIGHT_DECAY = USE_REGULARIZATION and REGULARIZATION_TYPE == "l2"
WEIGHT_DECAY = REGULARIZATION_FACTOR if USE_WEIGHT_DECAY else 0.0

GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]