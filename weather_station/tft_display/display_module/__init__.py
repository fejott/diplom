try:
    from .tft_display import TFTDisplay
    __all__ = ['TFTDisplay']
except ImportError:
    TFTDisplay = None
    __all__ = []
