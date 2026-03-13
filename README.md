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
  - `01_visualize_brats.ipynb`: Data visualization and exploration.
  - `02_dataloader_augmentation.ipynb`: Data loading and augmentation pipeline.
  - `03_spiking_transformer_training.ipynb`: Core Spiking Transformer training logic.
  - `04_llm_agentic_trial.ipynb`: Agentic triage using Phi-3-Mini.
  - `05_snn-with-agent-ipynb.ipynb`: Full training pipeline with 80/20 split and agent integration.
  - `06_evaluation_comparisons(dummy).ipynb`: Evaluation metrics and comparisons.
  - `07_evaluation_custom_data.ipynb`: Model evaluation on custom datasets.
- `models/`:
  - `spiking_unet_best.pth`: Best performing model checkpoint.
- `dataset/`:
  - `download.ipynb`: Utility notebook for downloading the BraTS 2023 dataset.
- `main.py`: Entry point for the Spiking Transformer scripts.
- `requirements.txt`: Project dependencies.
- `pyproject.toml`: Python project configuration file.
- `LICENSE`: MIT License file.

## Requirements

The project requires Python >= 3.12 and the following key dependencies:
- `torch`
- `snntorch`
- `langgraph`
- `transformers`
- `accelerate`
- `bitsandbytes`
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

To explore the pipeline, navigate to the `src/notebooks` directory and run the Jupyter notebooks in numerical order. These notebooks are optimized for high-performance computing environments like **Google Colab** and **Kaggle**, utilizing T4 GPU acceleration. 

> [!NOTE]
> For the agentic triage phase (04 and 05), the notebooks utilize 4-bit quantization via `bitsandbytes` to fit the Phi-3-Mini LLM within Colab's T4 RAM constraints.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
