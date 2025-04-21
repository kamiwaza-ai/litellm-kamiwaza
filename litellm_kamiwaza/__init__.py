import urllib3
import warnings

# Suppress InsecureRequestWarning for HTTPS requests to localhost
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from .kamiwaza_router import KamiwazaRouter
from .version import __version__

__all__ = ["KamiwazaRouter", "__version__"]
