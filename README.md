# Evaporate

<div align="center">
    <img src="assets/banner.png" alt="Evaporate diagram"/>
</div>

Code, datasets, and extended writeup for paper "Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes". 

## Setup

We encourage the use of conda environments:
```
conda create --name evaporate python=3.8
conda activate evaporate
```

Clone as follows:
```bash
# Evaporate code
git clone git@github.com:HazyResearch/evaporate.git
cd evaporate
pip install -r requirements.txt

# Weak supervision code
cd metal-evap
git submodule init
git submodule update
pip install -e .

# Manifest 
git clone git@github.com:HazyResearch/manifest.git
cd manifest
pip install -e .
```

## Datasets
The data used in the paper is hosted on HuggingFace's datasets platform: https://huggingface.co/datasets/hazyresearch/evaporate.

To download the datasets, run the following commands in your terminal:
```bash
git lfs install
git clone https://huggingface.co/datasets/hazyresearch/evaporate
```

Or download it via Python:
```python
from datasets import load_dataset
dataset = load_dataset("hazyresearch/evaporate")
```

The code expects the data to be stored at ``/data/evaporate/`` as specified in ``constants.py`` CONSTANTS, though can be modified.


## Running the code
Run closed IE and open IE using the commands:

```cd src/
bash run.sh
```

The ``keys`` in run.sh can be obtained by registering with the LLM provider. For instance, if you want to run inference with the OpenAI API models, create an account [here](https://openai.com/api/).


## Extended write-up
The extended write-up is included in this Github repository at [this URL](https://github.com/HazyResearch/evaporate/blob/main/technical-report.pdf).
