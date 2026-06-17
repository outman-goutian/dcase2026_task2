"""
Audio models module for the finetuning framework.

This module provides a unified interface for different pretrained audio models
(BEATs, AST, etc.) through the BaseAudioModel abstract class.
"""

__all__ = ['BaseAudioModel', 'BEATsModel', 'ASTAudioModel', 'EATAudioModel', 'DashengAudioModel']


def __getattr__(name):
    if name == 'BaseAudioModel':
        from .base_model import BaseAudioModel
        return BaseAudioModel
    if name == 'BEATsModel':
        from .beats_model import BEATsModel
        return BEATsModel
    if name == 'ASTAudioModel':
        from .ast_model import ASTAudioModel
        return ASTAudioModel
    if name == 'EATAudioModel':
        from .eat_model import EATAudioModel
        return EATAudioModel
    if name == 'DashengAudioModel':
        from .dasheng_model import DashengAudioModel
        return DashengAudioModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
