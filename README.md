# MHealth Data Analysis Project

## Overview
This project performs a comparative analysis of various machine learning models to classify human activities based on the MHealth dataset. The dataset contains sensor data (accelerometer and gyroscope readings) collected from multiple subjects performing different activities. The goal is to evaluate the performance of different models in accurately predicting activity types.

## Dataset
- **File**: `mhealth_raw_data.csv`
- **Description**: The dataset includes sensor measurements such as accelerometer (`alx`, `aly`, `alz`, `arx`, `ary`, `arz`) and gyroscope (`glx`, `gly`, `glz`, `grx`, `gry`, `grz`) readings, along with an `Activity` label and `subject` identifier.
- **Size**: 1,215,745 rows and 14 columns.
- **Features**:
  - `alx`, `aly`, `alz`: Accelerometer readings (left).
  - `glx`, `gly`, `glz`: Gyroscope readings (left).
  - `arx`, `ary`, `arz`: Accelerometer readings (right).
  - `grx`, `gry`, `grz`: Gyroscope readings (right).
  - `Activity`: Target variable (integer-encoded activity labels).
  - `subject`: Subject identifier (e.g., `subject1`, `subject10`).

## Project Structure
- **Main File**: `comparative_analysis.py` (or `Comparative_Analysis.ipynb`)
- **Purpose**: Implements data preprocessing, visualization, and model training/evaluation.
- **Sections**:
  1. **Data Loading**: Loads the MHealth dataset using pandas.
  2. **Preprocessing**: Checks for missing values, duplicates, and scales features.
  3. **Visualization**: Generates correlation heatmaps, histograms, and activity distribution plots.
  4. **Model Training and Evaluation**:
     - **K-Nearest Neighbors (KNN)**: Tests different neighbor values (1-9) and evaluates accuracy.
     - **Linear Regression**: Applies linear regression (not ideal for classification) and polynomial features.
     - **Support Vector Machine (SVM)**: Uses an RBF kernel for classification.
     - **Neural Network**: Implements a Keras-based sequential model with multiple dense layers.
     - **Logistic Regression**: Applies logistic regression for classification.
  5. **Evaluation Metrics**: Includes accuracy, precision, recall, F1-score, and confusion matrices for each model.

## Requirements
To run the project, install the required Python libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn keras
```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Prepare the Dataset**:
   - Place the `mhealth_raw_data.csv` file in the project directory or update the file path in the script.

3. **Run the Script**:
   - If using the `.py` file:
     ```bash
     python comparative_analysis.py
     ```
   - If using the Jupyter notebook:
     ```bash
     jupyter notebook Comparative_Analysis.ipynb
     ```

4. **Output**:
   - The script generates:
     - Data summaries (e.g., `df.describe()`, `df.info()`).
     - Visualizations (correlation heatmaps, histograms, count plots, confusion matrices).
     - Model performance metrics (accuracy, classification reports, cross-validation scores).

## Models and Performance
- **KNN**:
  - Evaluates different `n_neighbors` (1-9).
  - Outputs training, validation, and test accuracies, along with a classification report and confusion matrix.
- **Linear Regression**:
  - Applied with and without polynomial features.
  - Reports R² score, MSE, and RMSE.
  - Note: Linear regression is not suitable for this classification task.
- **SVM**:
  - Uses RBF kernel with `C=5` and `gamma=0.5`.
  - Reports validation and test accuracies, classification report, and confusion matrix.
- **Neural Network**:
  - Uses a Keras sequential model with dense layers (64, 128, 64, output).
  - Trained for 100 epochs with a batch size of 1024.
  - Reports test accuracy, training/validation accuracy, and loss plots.
- **Logistic Regression**:
  - Uses scikit-learn's `LogisticRegression`.
  - Reports validation and test accuracies, cross-validation scores, classification report, and confusion matrix.

## Notes
- The dataset is large, so a 10% subset is used for most models to reduce computational time.
- The `subject` column is dropped in some models as it is not relevant for activity classification.
- Ensure sufficient computational resources for training the neural network, as it requires significant memory and processing power.
- The Linear Regression section is included for demonstration but is not appropriate for this classification task.

## License
This project is licensed under the MIT License.
