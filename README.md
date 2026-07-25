# Telco Customer Churn Prediction — EDA & Predictive Modeling

## 📌 Project Objective

Customer churn is a critical problem in the telecommunications industry. Losing customers costs companies millions in lost revenue and acquisition costs.

The goal of this project is to:
- **Analyze** customer data to understand *why* customers leave.
- **Build** a Machine Learning model to predict which customers are at high risk of churning.
- **Provide** actionable business insights to improve customer retention.

---

## 📊 Dataset

**Source:** Telco Customer Churn Dataset on Kaggle

- **Rows:** 7,043 customer records
- **Columns:** 21 features
- **Target Variable:** `Churn` (Yes = customer left, No = customer stayed)

 Key Features

| Feature | Description |
| :--- | :--- |
| `tenure` | Months the customer has been with the company |
| `Contract` | Month-to-month, 1-year, or 2-year contract |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount paid |
| `InternetService` | DSL / Fiber optic / No internet |
| `PaymentMethod` | Electronic check, Mailed check, Bank transfer, Credit card |

---

🛠️ Tools & Libraries

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **Environment** | Jupyter Notebook / VS Code |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (Logistic Regression, Random Forest) |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |

---

📝 Step-by-Step Process

1. Data Cleaning
- Converted `TotalCharges` from `object` to numeric and filled missing values with the **median** to avoid skewing the data.
- Removed the `customerID` column (it provides no predictive value).
- Encoded the target variable `Churn` as binary (Yes = 1, No = 0).

2. Exploratory Data Analysis (EDA)
Performed visual analysis to uncover patterns:

- **Churn Rate:** ~26.5% of customers in the dataset churned.
- **Contract Type:** Month-to-month customers had a churn rate over **40%**, while 2-year contract customers had less than **5%** churn.
- **Tenure:** Customers with less than 12 months of tenure were significantly more likely to leave.
- **Payment Method:** Electronic check users showed higher churn compared to other payment methods.
- **Services:** Fiber optic internet users had slightly higher churn than DSL users.

3. Feature Engineering
- Created `TotalServices`: A count of how many additional services (security, backup, streaming, etc.) a customer subscribes to.
- Created `tenure_group`: Binned tenure into categories (0-12 months, 13-24 months, etc.) to capture non-linear trends.

4. Data Preprocessing
- Applied **One-Hot Encoding** to convert categorical variables into numeric format.
- Scaled numerical features using **StandardScaler** to ensure Logistic Regression performs optimally.

5. Handling Class Imbalance (SMOTE)
The dataset was imbalanced (74% stayed, 26% churned). If trained on raw data, the model would predict "Stayed" for everyone and fail to catch churners.

- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to balance the classes.
- This significantly improved the model's ability to identify actual churners (**Recall**).

6. Model Building
Two models were trained on the balanced dataset:
- **Logistic Regression** (Simple, interpretable baseline)
- **Random Forest Classifier** (Powerful ensemble model for non-linear patterns)

---

 📈 Results & Performance

| Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- |
| **Logistic Regression** | **75.6%** | **0.861** |
| **Random Forest** | **71.3%** | **0.827** |


- The relationships in this dataset (e.g., tenure, contract type) are **strongly linear**.
- Logistic Regression is a linear model—it draws a straight boundary and works perfectly when patterns are clear.
- Random Forest is more complex. With **default hyperparameters**, it can sometimes overfit or get lost in noise on smaller datasets (7,000 rows). 
- A ROC-AUC of **0.861** is considered **excellent** for churn prediction. It means the model is highly reliable in ranking risky customers.

---

Based on the EDA and Feature Importance analysis, the business should focus on:

1. **Convert Month-to-Month customers**: Offer incentives (discounts, free upgrades) to switch them to 1-year or 2-year contracts.
2. **Retain new customers**: Design onboarding campaigns for customers with tenure < 12 months to build loyalty early.
3. **Target high-risk payment methods**: Promote automatic payment methods (credit card/bank transfer) to electronic check users.
4. **Improve Fiber optic experience**: Address service quality or pricing for Fiber optic users, as they churn more than DSL users.

