import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump

# Load the data
df = pd.read_csv('Walmart.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Prepare the data by extracting date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['Is_Holiday'] = df['Holiday_Flag'].apply(lambda x: 'Holiday' if x == 1 else 'Non-holiday')

# Selecting relevant columns
df_selected = df[['Store', 'Year', 'Month', 'Week', 'Is_Holiday', 'Temperature', 'Weekly_Sales', 'Date']]

# Create training and testing sets
X = df_selected[['Store', 'Year', 'Month', 'Week', 'Is_Holiday', 'Temperature']]
y = df_selected['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Define preprocessing for numeric and categorical features
numeric_features = ['Store', 'Temperature']
numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
categorical_features = ['Year', 'Month', 'Week', 'Is_Holiday']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create a preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearRegression())])

# Fit the pipeline to train a regression model on the training set
model = pipeline.fit(X_train, y_train)

# Save the pipeline as a pickle file
dump(model, 'sales_forecasting_pipeline.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Create a DataFrame from X_test to keep 'Is_Holiday' information
X_test_with_predictions = X_test.copy()
X_test_with_predictions['Predicted_Sales'] = y_pred
X_test_with_predictions['Actual_Sales'] = y_test.reset_index(drop=True)

# Plotting predictions against actual sales with holiday highlighting
plt.figure(figsize=(10, 6))
for is_holiday, group in X_test_with_predictions.groupby('Is_Holiday'):
    plt.scatter(group['Actual_Sales'], group['Predicted_Sales'], alpha=0.5, label=f"{is_holiday}")
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Weekly Sales by Holiday')
plt.legend(title='Holiday')
plt.show()

# Plotting predictions against actual sales
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs. Predicted Weekly Sales')
plt.show()

# Plot a histogram of the weekly sales highlighting holidays
plt.figure(figsize=(10, 6))
for is_holiday, group in df_selected.groupby('Is_Holiday'):
    plt.hist(group['Weekly_Sales'], bins=30, alpha=0.7, label=f"{is_holiday}")
plt.title('Distribution of Weekly Sales by Holiday')
plt.xlabel('Weekly Sales')
plt.ylabel('Frequency')
plt.legend(title='Holiday')
plt.show()

# Plot a box plot of weekly sales by month
plt.figure(figsize=(10, 6))
df_selected.boxplot(column='Weekly_Sales', by='Month')
plt.title('Box Plot of Weekly Sales by Month')
plt.xlabel('Month')
plt.ylabel('Weekly Sales')
plt.show()

# Checking peak sales weeks and holidays
peak_weeks = df_selected[df_selected['Weekly_Sales'] >= df_selected['Weekly_Sales'].quantile(0.95)]
peak_weeks_holiday = peak_weeks[peak_weeks['Is_Holiday'] == 'Holiday']
print(peak_weeks_holiday[['Date', 'Weekly_Sales', 'Is_Holiday']])
