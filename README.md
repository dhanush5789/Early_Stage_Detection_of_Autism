# Early Stage Detection of Autism Spectrum Disorder (ASD) Using Decision Tree and Random Forest Algorithms

## Overview

This project aims to explore and implement machine learning algorithms, specifically Decision Tree and Random Forest, to detect the early stages of Autism Spectrum Disorder (ASD). Autism Spectrum Disorder is a complex neurodevelopmental condition characterized by difficulties in social interaction, communication, and restricted or repetitive behaviors. Early diagnosis is crucial for providing effective support and interventions, but detecting ASD can be challenging, especially in its early stages.

This repository contains code and data for building predictive models to identify early indicators of ASD based on various features and datasets. The project focuses on using two powerful classification algorithms — **Decision Tree** and **Random Forest** — to predict the likelihood of an individual being on the autism spectrum.

## Project Structure

```
early-stage-asd-detection/
│
├── data/                   # Contains dataset(s) used for training and testing the models
│   ├── upload.csv     # Example dataset of ASD-related features
│
├── src/                    # Source code for data preprocessing, model training, and evaluation
│   ├── preprocess.py       # Data preprocessing code (cleaning, feature selection, etc.)
│   ├── decision_tree.py    # Code for Decision Tree model training and evaluation
│   ├── random_forest.py    # Code for Random Forest model training and evaluation
│   ├── model_evaluation.py # Evaluation of models (accuracy, confusion matrix, etc.)
│
├── notebooks/              # Jupyter Notebooks for visualization and exploratory data analysis
│   ├── EDA.ipynb           # Exploratory Data Analysis (EDA) for understanding the dataset
│   ├── model_comparison.ipynb # Comparing Decision Tree and Random Forest performance
│
├── requirements.txt        # Python dependencies for the project
├── README.md               # Project documentation (this file)
```

## Data

The dataset used in this project contains features relevant to the detection of Autism Spectrum Disorder. These features include medical, behavioral, and demographic data points collected from a range of individuals. The dataset may contain:

- Behavioral traits and communication skills
- Response to social cues
- Family history of autism or other related conditions
- Demographic information like age, gender, etc.

Example data can be found in `data/autism_data.csv`.

## Setup Instructions

### Prerequisites

To run the project locally, ensure you have Python 3.10.9 installed. You'll also need to install the required dependencies.

### Install dependencies

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/early-stage-asd-detection.git
   cd early-stage-asd-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

1. **Data Preprocessing**  
   Preprocess the dataset (e.g., handle missing values, normalize the data) by running the following script:
   ```bash
   python src/preprocess.py
   ```

2. **Model Training**  
   To train the Decision Tree model:
   ```bash
   python src/decision_tree.py
   ```
   To train the Random Forest model:
   ```bash
   python src/random_forest.py
   ```

3. **Model Evaluation**  
   After training, evaluate the model’s performance using metrics such as accuracy, precision, recall, and F1-score:
   ```bash
   python src/model_evaluation.py
   ```

4. **Exploratory Data Analysis**  
   You can visualize the data and gain insights by opening the Jupyter notebook `notebooks/EDA.ipynb`.

5. **Model Comparison**  
   Compare the performance of the Decision Tree and Random Forest models in `notebooks/model_comparison.ipynb`.

## Algorithms Used

### Decision Tree

A Decision Tree is a supervised learning algorithm that splits the data into subsets based on the value of input features, resulting in a tree-like structure. The model is easy to interpret and visualize, making it a popular choice for classification tasks.

### Random Forest

Random Forest is an ensemble learning method that builds multiple Decision Trees and aggregates their predictions to improve classification accuracy. It helps reduce overfitting and improves model robustness by considering the diversity among individual trees.

## Evaluation Metrics

The following metrics will be used to evaluate the models:

- **Accuracy**: The proportion of correctly classified instances over the total instances.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

## Future Work

- **Hyperparameter Tuning**: The current models use default hyperparameters. Future work could involve optimizing these parameters for better performance.
- **Data Augmentation**: Expanding the dataset with additional features or more examples could improve the model's accuracy and generalizability.
- **Additional Algorithms**: Exploring other machine learning algorithms like SVM, Gradient Boosting, or Neural Networks might yield better results.


## Acknowledgments

- The dataset used in this project is available from publicly available sources. Please see the dataset documentation for more information.
- Thanks to the machine learning community for providing valuable resources and tutorials for Decision Trees and Random Forests.

---

Feel free to open an issue or pull request if you'd like to contribute, report bugs, or suggest improvements!

