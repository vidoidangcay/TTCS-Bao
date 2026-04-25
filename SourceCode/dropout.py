import torch.nn as nn
import config


def get_dropout():
    """Return a dropout layer based on config settings."""
    if config.USE_DROPOUT:
        return nn.Dropout(config.DROPOUT_PROB)
    return nn.Identity()
