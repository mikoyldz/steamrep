# Steam Review Sentiment Analysis Project

### Authors: Markus Grau, Daniel Kosma, Mikail Yildiz

## Overview
This project focuses on analyzing user reviews from the Steam platform to classify them as positive or negative using Natural Language Processing (NLP) techniques. The classification is based on the content of the written reviews, employing modern machine learning models like BERT (transformer) or classical machine learning approaches (such as Random Forest, or Logistic Regresion,...) for sentiment analysis.

## How to Run the Project

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- Pip package manager

### Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required packages by running:
    ```bash
    pip install -r requirements.txt
    ```

4. Find and open the `solution_wrapper_notebook.ipynb` which contains all the necessary steps to run the project and combine the functionalities of each module.

## Module Descriptions

### DataHandler Module
The `DataHandler` class manages all operations related to the raw review data. It includes methods for:
- Reading and loading the dataset,
- Preprocessing text data (e.g., cleaning, tokenization),
- Selecting the dataset size for training/testing.

To use it, import the class:

    ```python
    from DataHandler.DataHandler import DataHandler
    ```

### ModelPreparator Module
The `ModelPreparator` class handles the steps necessary to prepare the data for model training. It includes functionality such as:
- Generating emotion-based features (if applicable),
- Splitting the dataset into training and validation sets,
- Preparing data for input into the BERT model.

To use it, import the class:

    ```python
    from DataHandler.ModelPreparator import ModelPreparator
    ```

### ModelHandler Module
The `Bert_Trainer_Evaluator` class is responsible for training and evaluating the BERT model. It simplifies the process by combining training and evaluation into a single `.train` method, ensuring an efficient workflow.

To use it, import the class:

    ```python
    from model.ModelHandler import Bert_Trainer_Evaluator
    ```

## Running the Project
1. Install the dependencies as mentioned above.
2. Load the dataset into the appropriate location (if applicable).
3. Open and run the `solution_wrapper_notebook.ipynb`, which will guide you through the entire process, from data handling to model evaluation.

## Conclusion
This project provides an intuitive pipeline for classifying Steam reviews as positive or negative, leveraging state-of-the-art NLP techniques with BERT. Each module is designed to handle specific aspects of the pipeline, from data handling to model evaluation, making it easy to extend or modify for future improvements.

