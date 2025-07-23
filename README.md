# 🚗 Customer Purchase Prediction

This project predicts whether a customer will purchase a **car** based on their **age** and **salary** using multiple machine learning classification models.

## 📌 Project Overview

Given a customer's **age** and **estimated salary**, the models predict whether they are likely to purchase a car (`Will Buy` or `Will Not Buy`). This is a classic **binary classification** task built using the **Social Network Ads** dataset.

## ✅ Models Implemented

The following machine learning models have been trained and evaluated:

* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM - Linear Kernel)**
* **Kernel SVM (RBF Kernel)**
* **Naive Bayes**
* **Decision Tree Classifier**
* **Random Forest Classifier**

## 🧠 Input Features

* `Age`: Customer's age
* `EstimatedSalary`: Estimated annual income of the customer

## 🎯 Output

* **Prediction**: Whether the customer will buy the product
* **Accuracy**: Displayed beside each model to indicate performance

## 🛠️ Tech Stack

* Python
* pandas, NumPy
* scikit-learn
* matplotlib & seaborn (for visualization)
* joblib (for saving models)

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Purchase-Prediction.git
   cd Purchase-Prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python app.py
   ```

4. Enter `Age` and `Estimated Salary` to get predictions from each model.

## 📊 Example

```python
Input: Age = 35, Salary = 60,000
Output:
Logistic Regression: Will Not Buy (Accuracy: 89%)
KNN: Will Not Buy (Accuracy: 93%)
...
```

## 📈 Evaluation

Each model was evaluated on a test set using accuracy. Further performance metrics (like confusion matrix, precision, recall, etc.) can also be added for deeper insights.

## 📁 Dataset

The dataset used is based on the popular **Social_Network_Ads.csv**, commonly used for binary classification demos.
