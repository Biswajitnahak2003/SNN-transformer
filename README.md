# Spiking Transformer U-Net for BraTS 2023

This repository contains the code and Jupyter Notebooks for training a Spiking Transformer U-Net architecture on the BraTS 2023 dataset. The project implements a comprehensive 6-phase pipeline that focuses on memory efficiency and advanced triage methodologies.

## Pipeline Phases

1. **Data Ingestion**: Loading and preparing the BraTS 2023 MRI dataset.
2. **Spiking Tokenization**: Converting continuous input data into spikes.
3. **Agentic Triage**: Utilizing entropy and a Phi-3-Mini Large Language Model (LLM) for processing and triage.
4. **Spiking Transformer Encoder**: Encoding spatial-temporal features using a Spiking Transformer.
5. **Spiking U-Net Decoder**: Reconstructing the segmentation masks.
6. **Forward Propagation Through Time (FPTT) Optimization**: Enhancing memory efficiency during training.

## Project Structure

- `src/notebooks/`: Contains the Jupyter notebooks for the different phases of training and evaluation.
  - `01_visualize_brats.ipynb`
  - `02_dataloader_augmentation.ipynb`
  - `03_spiking_transformer_training.ipynb`
  - `04_llm_agentic_triage.ipynb`
  - `05_full_training.ipynb`
  - `06_evaluation_comparisons.ipynb`
  - `07_evaluation_custom_data.ipynb`
- `main.py`: Entry point for the Spiking Transformer scripts.
- `requirements.txt`: Project dependencies.
- `pyproject.toml`: Python project configuration file.
- `LICENSE`: MIT License file.

## Requirements

The project requires Python >= 3.12 and the following key dependencies:
- `torch`
- `snntorch`
- `langgraph`
- `nibabel`
- `numpy`
- `pandas`
- `matplotlib`
- `jupyter`
- `ipykernel`

## Installation

You can install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Setup & Usage

To explore the pipeline, navigate to the `src/notebooks` directory and run the Jupyter notebooks in numerical order. These notebooks are designed to run in environments like Google Colab with GPU acceleration.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
