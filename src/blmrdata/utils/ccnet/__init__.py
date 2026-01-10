# -*- coding: utf-8 -*-
"""CCNet utilities for perplexity-based text filtering.

This module provides utilities for downloading and using CCNet language models
for perplexity estimation, which can be used to filter text datasets.

Example usage for downloading models:
    >>> from blmrdata.utils.ccnet import download_perplexity_models
    >>> download_perplexity_models(languages=["fr"], output_folder="data/perplexity_models")

Example usage for computing perplexity:
    >>> from blmrdata.utils.ccnet import Perplexity
    >>> perplexity_model = Perplexity("data/perplexity_models", "fr")
    >>> score = perplexity_model.pp("Some text to evaluate")
    >>> print(f"perplexity: {1 / score:.2f}")
"""

from blmrdata.utils.ccnet.perplexity import (
    download_perplexity_models,
    download_language_models,
    Perplexity,
)

__all__ = [
    "download_perplexity_models",
    "download_language_models",
    "Perplexity",
]
