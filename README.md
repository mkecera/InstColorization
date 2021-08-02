# Image Colorization Group Project

## Installation

In order to setup a local environment, create a venv and run `pip install -r requirements.txt` in the root of the project

## How to run on Google Colab

The results can be reproduced by running the notebooks placed under `/notebooks`.

The notebooks can be opened in `Colab` and self-contain the environment installation

### Notebooks

- `train_and_validate_default.ipynb` Runs the default model and produces plots and csv with results
- `train_and_validate_larger_fusion_network.ipynb` Runs the extended fusion model and produces plots and csv with results 
- `train_and_validate_more_weight.ipynb` Runs the fusion model with extended weights and produces plots and csv with results
- `train_and_validate_smaller_fusion_network.ipynb` Runs the reduced fusion model and produces plots and csv with results 

### File Structure

- `train_with_validation.py` Trains and validates the model
- `test_fusion.py` Tests the images using a test image set
- `inference_bbox.py` Creates bounding boxes of a dataset
- `calculate_metrics.py` Calculates metrics for a test image set
- `/models` Includes the different models of our implementation