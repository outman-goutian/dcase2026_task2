import yaml
from types import SimpleNamespace
from pathlib import Path
from functools import partial


def load_config(config_path: str = None):
    """
    加载 YAML 配置文件并返回嵌套的命名空间对象

    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径

    Returns:
        嵌套的 SimpleNamespace 对象，支持 config.data.xxx 方式访问
    """
    if config_path is None:
        config_path = "config.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_namespace(d):
        """递归将字典转换为 SimpleNamespace"""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d

    config = dict_to_namespace(config_dict)

    # 添加 to_dict 方法
    def to_dict(obj):
        """将 namespace 转换回字典"""
        if isinstance(obj, SimpleNamespace):
            return {k: to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        else:
            return obj

    config.to_dict = partial(to_dict, config)

    return config