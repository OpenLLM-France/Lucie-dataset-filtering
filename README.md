# Bloom-ng-dataset-processing - Data processing scripts Bloomer-7B

Code to compile and process data from Bloomer-7B experiment

## Installation

requirements: python 3.8+

```bash
cd Bloomer-dataset-processing
pip install -e ./
```

## Usage

```bash
blmrdata --help
```
### I- Perplexity Inference

#### 1- Downloading the models

```bash
blmrdata download perplexity-models --output-folder path/to/model/download/folder
```
#### 2- Inference on a text

```python

from blmrdata.utils.ccnet.perplexity import Perplexity

pp_compute = Perplexity(
    model_download_folder = "path/to/model/download/folder",
    language = "en"
    )

text = "This is a test text."

print("Perplexity: ", pp_compute(text))
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
