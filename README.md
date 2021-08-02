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

## ConvLSTM extension

### Training
1. Put the training data (all of your videos) under video_module/data/training.  
2. Run this command in order to extract the frames from your videos and build the training files structure:
```
python3 video_modules/data_preprocessing.py
```
3. Run this command in order to extract the inference bounding boxes from all the frames.
```
python3 video_modules/inference_bbox.py
```
4. Run this command in order to get the output for all the frames from instance colorization approach.
```
python3 test_fusion_for_video_module.py
```
5. Run this command to train the ConvLSTM model.
```
python3 lstm_main.py --train
```

### Testing
1. You need to [download](https://drive.google.com/drive/folders/1r3PoTE9K0iX-7K56xt8AI6SKlBBANQ-c?usp=sharing) the model checkpoint and to put in under video_modules/models/
2. Repeat training steps from 1 to 4 under training section while replacing the training data with the testing one.
3. Run this command to get the predictions
```
python3 lstm_main.py
```