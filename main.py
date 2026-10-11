
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv( r"C:\Users\HP\Desktop\ML\Attrition\WA_Fn-UseC_-HR-Employee-Attrition.csv")
# Display first few rows
print(df.head())
# Drop irrelevant columns
df.drop(
    ['EmployeeNumber', 'Over18', 'StandardHours'],
    axis=1,
    inplace=True
)
# Separate features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition'].map({'No': 0, 'Yes': 1})
# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         categorical_columns)
    ],
    remainder='passthrough'
)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
# Create the XGBoost pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test,y_pred,target_names=['No Attrition', 'Attrition']))
