import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
file_path = "E:\\busi\\telco.csv"
df = pd.read_csv(file_path)

# Convert 'TotalCharges' to numeric (handling empty values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# Convert target column to binary
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Define feature set
selected_features = ['tenure', 'InternetService', 'StreamingTV', 'StreamingMovies', 
                     'TechSupport', 'Contract', 'MonthlyCharges', 'TotalCharges']
target = 'Churn'

# Split dataset before encoding
X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df[target], test_size=0.2, random_state=42)

# Define preprocessing (Encoding + Scaling)
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['tenure', 'MonthlyCharges', 'TotalCharges']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['InternetService', 'StreamingTV', 'StreamingMovies', 'TechSupport', 'Contract'])
])

# Apply preprocessing first
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Apply SMOTE only after encoding
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train the model
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Save the new model and preprocessor
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(preprocessor, open('preprocessor.pkl', 'wb'))
