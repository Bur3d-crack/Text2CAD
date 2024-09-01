# File: README.md
# Text-to-CAD Model using Point Clouds

This project aims to create a text-to-CAD model using point cloud data as an intermediate representation. It uses PyTorch to train a model that generates point clouds from text descriptions.

## Project Structure

- `main.py`: The main script to run the model
- `model.py`: Contains the PyTorch model definition
- `dataset.py`: Defines the custom dataset for text and point cloud data
- `utils.py`: Utility functions for data preprocessing and point cloud handling
- `requirements.txt`: List of Python dependencies

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your dataset of text descriptions and point clouds
4. Run the training script: `python main.py`

## TODO

- Implement point cloud to CAD QUERY conversion
- Add evaluation metrics
- Optimize model architecture and hyperparameters
