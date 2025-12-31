from yacs.config import CfgNode as CN
from match_anything.config.default import get_cfg_defaults

def load_config(method):
    """
    Load the config files for the corresponding MatchAnything model

    :param method: name of the MatchAnything model, must be either `matchanything_roma` or `matchanything_eloftr`
    """
    def lower_config(yacs_cfg):
        if not isinstance(yacs_cfg, CN):
            return yacs_cfg
        return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

    config = get_cfg_defaults()
    if method == 'matchanything_eloftr':
        config.merge_from_file("configs/models/eloftr_model.py")
    elif method == 'matchanything_roma':
        config.merge_from_file("configs/models/roma_model.py")
    else:
        raise ValueError(f"Method {method} not recognized. Supported methods are: ROMA, ELoFTR")

    config.METHOD = method
    config = lower_config(config)

    if method == 'matchanything_roma':
        return config['roma']
    elif method == 'matchanything_eloftr':
        return config['loftr']