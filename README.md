# Rice Classification using Different Models

## Project Overview
This project focuses on classifying two types of rice grains, Cammeo and Osmancık, using various machine learning models. The goal is to accurately distinguish between these two varieties based on their morphological features.

## Dataset
The dataset used for this classification task contains morphological features of Cammeo and Osmancık rice grains. Each rice grain is described by the following attributes:

*   **Area**: Number of pixels within the boundaries of the rice grain.
*   **Perimeter**: Circumference of the rice grain.
*   **Major Axis Length**: Longest line that can be drawn on the rice grain.
*   **Minor Axis Length**: Shortest line that can be drawn on the rice grain.
*   **Eccentricity**: Measures how round the ellipse, which has the same moments as the rice grain, is.
*   **Convex Area**: Pixel count of the smallest convex shell of the region formed by the rice grain.
*   **Extent**: Ratio of the region formed by the rice grain to the bounding box pixels.
*   **Class**: The type of rice, either 'Cammeo' or 'Osmancik'.

The dataset was sourced from Kaggle: `muratkokludataset/rice-dataset-commeo-and-osmancik`.

## Methodology
The project follows a standard machine learning workflow:

1.  **Data Loading and Initial Exploration**: The dataset was loaded, and initial checks were performed for missing values and duplicates. Basic statistical summaries and data types were inspected.
2.  **Exploratory Data Analysis (EDA)**: Visualizations such as pair plots, box plots, histograms, and distribution plots were generated to understand the data distribution, relationships between features, and potential outliers. A correlation matrix was also plotted to identify feature relationships.
3.  **Data Preprocessing**: 
    *   The 'Class' column (target variable) was converted into numerical format using `pd.get_dummies`.
    *   The dataset was split into training (75%) and testing (25%) sets.
    *   Feature scaling was applied using `StandardScaler` to normalize the numerical features.
    *   The target variable `y_train` and `y_test` were converted to 1D arrays (`.ravel()`) to fit scikit-learn's expected input format.
4.  **Model Training and Hyperparameter Tuning**: Four different classification models were trained and tuned using `GridSearchCV` to find the best hyperparameters:
    *   **Decision Tree Classifier** (`max_depth`, `min_samples_leaf`)
    *   **Random Forest Classifier** (`n_estimators`, `max_depth`)
    *   **K-Nearest Neighbors Classifier** (`n_neighbors`)
    *   **Logistic Regression** (`C`)

5.  **Model Evaluation**: Each trained model was evaluated on the test set using the following metrics:
    *   Accuracy Score
    *   Classification Report (Precision, Recall, F1-score)
    *   Confusion Matrix

## Results and Best Performing Model

After training and evaluating all models with their optimized hyperparameters, the following accuracies were observed on the test set:

*   **Decision Tree**: 0.913
*   **Random Forest**: 0.929
*   **K-Nearest Neighbors**: 0.923
*   **Logistic Regression**: **0.931**

**Logistic Regression** emerged as the best-performing model, achieving an accuracy of approximately 93.1%. Its performance was consistently strong across both 'Cammeo' and 'Osmancik' classes, with high precision, recall, and F1-scores. This suggests that the features provide sufficient linear separability for effective classification of the two rice varieties.

## Dependencies
The following Python libraries are required to run this notebook:

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `kagglehub`

These can be installed using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

## Usage
To run this notebook:

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd rice-classification
    ```
2.  Install the required dependencies (as listed above).
3.  Open and run the `rice_classification.ipynb` notebook in a Jupyter environment (e.g., Google Colab, Jupyter Notebook, JupyterLab).

## Conclusion
This project successfully demonstrates the application of various machine learning algorithms for rice grain classification. The Logistic Regression model proved to be highly effective, providing accurate differentiation between Cammeo and Osmancik rice based on their morphological features.
