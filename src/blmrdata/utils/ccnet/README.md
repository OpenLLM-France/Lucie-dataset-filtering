# CCNet Perplexity Library

A Python library for text quality assessment using perplexity-based filtering with CCNet language models.

## Overview

This library provides utilities for downloading and using CCNet language models to compute perplexity scores for text documents. Perplexity is a measure of how well a language model predicts a text - lower perplexity indicates text that is more similar to the training corpus, typically corresponding to higher quality, well-formed text.

## Features

- **Automatic model downloading** from Facebook's CCNet repository
- **Multi-language support** (French, English, and many other languages)
- **Perplexity computation** using KenLM language models
- **SentencePiece tokenization** for accurate text processing
- **Text normalization** utilities for preprocessing

## Installation

This library is part of the `blmrdata` package. Install the parent package:

```bash
cd Lucie-dataset-filtering
pip install -e ./
```

### Dependencies

- `kenlm` - Language model toolkit
- `sentencepiece` - Tokenization library
- `loguru` - Logging
- `tqdm` - Progress bars

## Quick Start

### Basic Usage

```python
from pathlib import Path
from blmrdata.utils.ccnet import download_ccnet_models, Perplexity

# Setup paths and languages
cc_net_model_path = Path("assets/ccnet_models")
languages = ["fr"]

# Download perplexity models for French
download_ccnet_models(languages=languages, output_folder=cc_net_model_path)

# Initialize the perplexity model
perplexity_model = Perplexity(cc_net_model_path, languages[0])

# Compute perplexity for a text
text = "Ceci est un exemple de texte en français."
score = perplexity_model.pp(text)
# Lower perplexity score (higher value when inverted) indicates better quality
print(f"Perplexity: {1 / score:.2f}")
```

## API Reference

### Functions

#### `download_ccnet_models(languages: List[str], output_folder: Path) -> None`

Download perplexity models for specified languages.

**Parameters:**
- `languages` (List[str]): List of language codes (e.g., `['fr', 'en']`)
- `output_folder` (Path): Destination folder for downloaded models

**Example:**
```python
download_ccnet_models(languages=["fr", "en"], output_folder=Path("models"))
```

#### `download_language_models(language: str, output_folder: Path) -> Tuple[Path, Path]`

Download ARPA and SentencePiece models for a single language.

**Parameters:**
- `language` (str): Language code (e.g., `'fr'`, `'en'`)
- `output_folder` (Path): Destination folder

**Returns:**
- Tuple[Path, Path]: Paths to the ARPA and SentencePiece model files

### Classes

#### `Perplexity`

Class to compute perplexity scores for text documents.

**Constructor:**
```python
Perplexity(model_download_folder: Path, language: str)
```

**Parameters:**
- `model_download_folder` (Path): Path to folder containing downloaded models
- `language` (str): Language code (e.g., `'fr'`, `'en'`)

**Methods:**

##### `pp(text: str, normalize: bool = False) -> float`

Compute the perplexity of a text.

**Parameters:**
- `text` (str): Text to evaluate
- `normalize` (bool): Whether to normalize text before computing perplexity (default: False)

**Returns:**
- float: Perplexity value (10^(-average_log_prob))

**Example:**
```python
perplexity_model = Perplexity("data/models", "fr")
score = perplexity_model.pp("Bonjour le monde!")
print(f"Perplexity: {1 / score:.2f}")
```


## Supported Languages

The library supports all languages available in Facebook's CCNet repository. Common languages include:

- `fr` - French
- `en` - English
- `de` - German
- `es` - Spanish
- `it` - Italian
- `pt` - Portuguese
- `nl` - Dutch
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ar` - Arabic

For a complete list, see the [Facebook CCNet repository](https://github.com/facebookresearch/cc_net),
for instance in the [Makefile](https://github.com/facebookresearch/cc_net/blob/main/Makefile#L10).

## Understanding Perplexity Scores

### What is Perplexity?

Perplexity is a measurement of how well a probability model predicts a sample. In the context of language models:

- **Lower perplexity** = Better prediction = Text is more similar to training data
- **Higher perplexity** = Worse prediction = Text is less similar to training data

### Interpreting Scores

The `pp()` method returns a value where:
- Lower values indicate **higher quality** text (more similar to well-formed language)
- Higher values indicate **lower quality** text (unusual patterns, errors, noise)

To get the traditional perplexity metric (where higher = worse), use:
```python
traditional_perplexity = 1 / perplexity_model.pp(text)
```

### Use Cases

Perplexity filtering is useful for:
- **Dataset quality control** - Remove low-quality documents
- **OCR error detection** - Identify poorly scanned documents
- **Language identification** - Verify text is in expected language
- **Content filtering** - Separate formal vs. informal text
- **Preprocessing** - Rank documents by quality before training

## Advanced Usage

### Batch Processing

```python
from pathlib import Path
from blmrdata.utils.ccnet import Perplexity

# Initialize model
perplexity_model = Perplexity(Path("data/models"), "fr")

# Process multiple documents
documents = [
    "Premier document de haute qualité.",
    "Deuxième document avec du texte.",
    "Troisième document à évaluer."
]

results = []
for doc in documents:
    score = perplexity_model.pp(doc)
    results.append({
        "text": doc,
        "perplexity": 1 / score,
        "quality_score": score
    })

# Sort by quality (higher score = better quality)
results.sort(key=lambda x: x["quality_score"], reverse=True)

for r in results:
    print(f"Perplexity: {r['perplexity']:.2f} - {r['text'][:50]}...")
```

### Using Text Normalization

```python
# Compute perplexity with text normalization
score_normalized = perplexity_model.pp(text, normalize=True)

# Normalization includes:
# - Lowercasing
# - Accent removal
# - Unicode punctuation replacement
# - Non-printing character removal
```

### Getting Detailed Metrics

```python
# Get both perplexity and text length
avg_neg_log_prob, length = perplexity_model(text)

print(f"Average negative log probability: {avg_neg_log_prob:.4f}")
print(f"Text length (tokens): {length}")
print(f"Perplexity: {10.0 ** -avg_neg_log_prob:.2f}")
```

## Module Structure

```
ccnet/
├── __init__.py              # Main exports
├── perplexity.py            # Core perplexity computation
├── README.md                # This file
└── ccnet_utilities/
    ├── __init__.py
    ├── sentencepiece.py     # SentencePiece tokenization
    └── text_normalizer.py   # Text normalization utilities
```

## Technical Details

### Model Files

For each language, two files are downloaded:
- `{lang}.arpa.bin` - KenLM language model (ARPA format)
- `{lang}.sp.model` - SentencePiece tokenization model

### Processing Pipeline

1. **Tokenization**: Text is tokenized using SentencePiece
2. **Scoring**: Each line is scored using the KenLM language model
3. **Aggregation**: Scores are aggregated across all lines
4. **Perplexity**: Final perplexity is computed as 10^(-avg_log_prob)

## References

- **CCNet**: [Facebook's CCNet Repository](https://github.com/facebookresearch/cc_net)
- **KenLM**: [Language Model Toolkit](https://kheafield.com/code/kenlm/)
- **SentencePiece**: [Unsupervised Text Tokenizer](https://github.com/google/sentencepiece)
- **Lucie Dataset**: [OpenLLM-France/Lucie-Training-Dataset](https://huggingface.co/datasets/OpenLLM-France/Lucie-Training-Dataset)

## License

MIT License

## Contact

- **Organization**: OpenLLM-France
- **Email**: contact@openllm-france.fr
- **Repository**: [OpenLLM-France/Lucie-dataset-filtering](https://github.com/OpenLLM-France/Lucie-dataset-filtering)

## Contributing

Contributions are welcome! Please see the main repository's [CONTRIBUTING.md](../../../../CONTRIBUTING_Lucie-dataset-filtering.md) for guidelines.

