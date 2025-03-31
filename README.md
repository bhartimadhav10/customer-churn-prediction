# Customer Churn Prediction using Artificial Neural Networks

## Project Overview
This project aims to predict customer churn using an Artificial Neural Network (ANN). The dataset used for training and evaluation is stored in a CSV file located in the same directory as the Jupyter Notebook.

## Dataset
The dataset is in CSV format and contains customer details, including demographics, account information, and whether they have churned. The dataset is preprocessed before training the model. Kindly update the dataset path as per your file settings.

## Steps in the Notebook

### 1. Importing Libraries
The project uses Python libraries such as:
- NumPy
- Pandas
- TensorFlow
- Scikit-learn
- Matplotlib

### 2. Data Preprocessing
- The dataset is read using Pandas.
- Categorical variables such as `Gender` and `Geography` are encoded using Label Encoding and One-Hot Encoding.
- The data is split into training and test sets.
- Feature scaling is applied to standardize the dataset.

### 3. Building the ANN
- A sequential model is created using TensorFlow.
- The ANN consists of input, hidden, and output layers.
- The activation functions used are `ReLU` for hidden layers and `sigmoid` for the output layer.
- The model is compiled with an `adam` optimizer and `binary_crossentropy` loss function.

### 4. Training the ANN
- The model is trained on the training dataset.
- Performance metrics such as accuracy are used to evaluate the model.

### 5. Making Predictions
- The model predicts customer churn on test data.
- A confusion matrix and accuracy score are used to assess performance.

## Usage
To run the project:
1. Ensure all required libraries are installed.
2. Open and run the Jupyter Notebook.
3. The dataset should be placed in the same directory as the notebook.

## Dependencies
Install the required dependencies using:
```bash
pip install -r requirements.txt
```
### Required Libraries:
- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Project Structure
```
customer-churn-prediction/
│── dataset.csv                # Customer churn dataset
│── churn-customer.ipynb        # Jupyter Notebook with ANN model
│── requirements.txt            # Project dependencies
│── .gitignore                  # Files to ignore
│── README.md                   # Project description
```

## Conclusion
This project demonstrates the application of Artificial Neural Networks in predicting customer churn. Proper data preprocessing and feature engineering significantly improve model accuracy.