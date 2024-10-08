# -*- coding: utf-8 -*-
# flake8: noqa: F401

from .bark import BarkDeploy
from .chattts import ChatTTSDeploy
from .musicgen import MusicGenDeploy
from .base import TTSDeploy

__all__ = [
    'TTSDeploy'
    'BarkDeploy',
    'ChatTTSDeploy',
    'MusicGenDeploy',
]
