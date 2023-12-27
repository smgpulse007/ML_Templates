import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Replace these with your actual categorical and numerical columns
categorical_cols = [col for col in df_premod.columns if df_premod[col].dtype == 'object']
numerical_cols = [col for col in df_premod.columns if df_premod[col].dtype in ['int64', 'float64']]

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessing for numerical data
numerical_transformer = RobustScaler()

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = XGBClassifier()
# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])
# Define your target variable
y = df_premod['LOS_2_categories'] # replace 'LOS' with your actual target column name
X = df_premod.drop(['LOS_2_categories'], axis=1)

label_mapping = {'0-7d': 0, '8+d': 1}
y = y.map(label_mapping)

# Split data into train and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = {i: class_weights[i] for i in range(len(class_weights))}

# Map each y value to its weight
sample_weights = y_train.map(weights)

# Fit the model
# Fit the model with sample weights correctly passed
clf.fit(X_train, y_train, model__sample_weight=sample_weights)

# Preprocessing of validation data, get predictions
predictions = clf.predict(X_test)

print(classification_report(y_test, predictions))

