
# 🏠 Housing Price Prediction using Linear Regression

This project demonstrates how to build a **Linear Regression** model to predict **housing prices** based on various features.

---

## 📂 Dataset

The dataset used (`Housing.csv`) contains multiple features about properties such as:
- Area (in square feet)
- Number of bedrooms
- Number of bathrooms
- Parking spaces
- Furnishing status (`furnished`, `semi-furnished`, `unfurnished`)
- Presence of amenities (`yes`/`no`)
- And the target: **Price**

---

## 🚀 Project Workflow

### 1. Importing Libraries
Essential Python libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`) are imported for data handling, modeling, and evaluation.

### 2. Loading and Preprocessing the Dataset
- Dataset is loaded using `pandas`.
- Categorical values like `'yes'` and `'no'` are replaced with `1` and `0`.
- Categorical features (like `furnishingstatus`) are encoded using **One-Hot Encoding**.
- Null values are checked to ensure data cleanliness.

### 3. Splitting the Data
- The dataset is split into **training** (80%) and **testing** (20%) sets.
- `train_test_split` from `scikit-learn` is used to ensure randomness.

### 4. Training the Linear Regression Model
- A **Linear Regression** model is built using `LinearRegression()` from `sklearn.linear_model`.
- The model is trained on the training set.

### 5. Model Evaluation
- The model is evaluated on the test set using:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R-squared (R² Score)**

### 6. Visualization
- A scatter plot is created showing **Actual vs Predicted** housing prices.
- A perfect prediction would lie on the red dashed line in the plot.

### 7. Interpretation of Coefficients
- The impact of each feature on house price prediction is analyzed by examining the model's **coefficients** and **intercept**.

---

## 📈 Results

The model provides a baseline understanding of how different factors influence housing prices and evaluates its predictive performance using standard regression metrics.

---

## 🛠 Technologies Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📚 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repository
   ```
3. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Run the Python script.

---

## 📌 Note
- Ensure `Housing.csv` file is placed in the correct directory or update the file path accordingly.
- Categorical columns must be encoded properly before training machine learning models.

---

## ✨ Author
-Abhishek Singh



