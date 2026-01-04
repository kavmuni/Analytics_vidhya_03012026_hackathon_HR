import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score, precision_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib as jb
import streamlit as st

hr_train_df = pd.read_csv('../Dataset/train.csv')
hr_test_df = pd.read_csv('../Dataset/test.csv')
hr_test_df_copy=hr_test_df.copy()

# convert all the values of both data frame into upper case so all the values are treated same when it comes to unknown data
for df in [hr_train_df, hr_test_df]:
    df.rename(columns={"KPIs_met >80%": "KPIs_met_gt_80"}, inplace=True)
    df.rename(columns={"awards_won?": "awards_won"}, inplace=True)
    df.drop_duplicates()
    df.drop(columns=['employee_id'], inplace=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.upper()
    df['education'] = df['education'].fillna(df.groupby(['gender', 'department'])['education'].transform(lambda x: x.mode()[0]))
    df['education'] = df['education'].fillna(df.groupby(['gender'])['education'].transform(lambda x: x.mode()[0]))
    df['previous_year_rating'] = df['previous_year_rating'].fillna(df.groupby(['gender', 'department'])['previous_year_rating'].transform(lambda x: x.mode()[0]))
    df['previous_year_rating'] = df['previous_year_rating'].fillna(df.groupby(['gender'])['previous_year_rating'].transform(lambda x: x.mode()[0]))

X = hr_train_df.drop(columns=['is_promoted'])
y = hr_train_df['is_promoted']

num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), #Imputer is NOT needed here as the NULL values are filled above
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), #Imputer is NOT needed here as the NULL values are filled above
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

xgb_classifier = XGBClassifier(random_state=42, n_estimators=200, max_depth=8, eta=0.1, max_delta_step=2)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# It's better to explicitly use the xgb_classifier in the pipeline if that's the model to be tuned
model_pipeline=Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_classifier)
    ]
)
""" Hyperparameter Tuning using GridSearchCV """
"""
grid_params = [
    {
        # "model": [xgb_classifier] is not strictly necessary here if 'model' step is already xgb_classifier
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [7, 8, 9],
        "model__eta": [0.01, 0.1, 0.2],
        #"model__scale_pos_weight": np.linspace(1,2,num=50).tolist() # Corrected: pass individual float values
        "model__max_delta_step": [0, 1, 2]
    }
]

grid_1 = GridSearchCV(estimator=model_pipeline, param_grid=grid_params, cv=5, n_jobs=-1, verbose=1, scoring="f1")
grid_1.fit(X_train, y_train)
print(grid_1.best_params_)
"""
model_pipeline.fit(X_train, y_train)
print("***************Train Data Set**********************")
y_train_pred = model_pipeline.predict(X_train)
print("Train F1 Score:", f1_score(y_train, y_train_pred))
print("Train Recall Score:", recall_score(y_train, y_train_pred))
print("Classification Report:\n", classification_report(y_train, y_train_pred))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("***************Validation Data Set**********************")
y_val_pred = model_pipeline.predict(X_val)
print("Validation F1 Score:", f1_score(y_val, y_val_pred))
print("Validation Recall Score:", recall_score(y_val, y_val_pred))
print("Classification Report:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

accuracy_score = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

jb.dump(model_pipeline, '../model/xgb_hr_model.pkl')
print("Model saved successfully.")

