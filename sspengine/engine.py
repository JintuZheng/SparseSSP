from mmengine.config import Config
from mmengine.registry import Registry

__all__ = ['BUILDER', 'MAP_FUNC', 'Config']

BUILDER = Registry('builder')
MAP_FUNC = Registry('map_fn')
Config = Config


