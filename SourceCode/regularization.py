import config


def get_regularization_type():
    return config.REGULARIZATION_TYPE.strip().lower() if hasattr(config, "REGULARIZATION_TYPE") else "none"


def get_regularization_factor():
    return config.REGULARIZATION_FACTOR if getattr(config, "USE_REGULARIZATION", False) else 0.0


def get_l2_weight():
    return get_regularization_factor() if get_regularization_type() == "l2" else 0.0


def get_l1_weight():
    return get_regularization_factor() if get_regularization_type() == "l1" else 0.0
