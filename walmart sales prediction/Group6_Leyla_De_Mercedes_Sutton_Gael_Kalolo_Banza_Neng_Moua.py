import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from joblib import dump

# filtering the data and fixing the data to the same format
df = pd.read_csv("Walmart.csv")

df_selected = df[['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature']]
df_selected.to_csv('Walmart2.csv', index=False)

df1 = pd.read_csv("Walmart2.csv")

# replacing blank space with NaN values in order to drop them
df1 = df1.replace('', pd.NA)
df1 = df1.dropna()

print(df1.head(10))  # printing the top 10 row from the CSV file

# creating a histograms
data = df1[['Store', 'Weekly_Sales', 'Holiday_Flag', 'Temperature']].values

# Create histograms for each column
for i in range(data.shape[1]):
    plt.hist(data[:, i], bins=25, density=True, alpha=0.5, color='#86bf91', zorder=2, rwidth=0.9)
    plt.ylabel('Density')
    plt.title('Histogram of Walmart')
    plt.show()

# creating a logistic model
# train the model
# separate features and labels
features = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature']
label = 'Store'
X, y = df1[features].values, df1[label].values

# creating the train and test of the data by splitting in a 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
reg = 0.01

# define preprocessing for numeric columns
numeric_features = [0, 2, 3, 4]
numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

# define preprocessing for categorical feature
categorical_features = [1]
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# creating a preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('logregressor', LogisticRegression(C=1 / reg, solver="liblinear"))])

# fit the pipeline to train a logistic regression model on the training set
model = pipeline.fit(X_train, y_train)

# Save the pipeline as a pickle file
dump(pipeline, 'logistic_regression_pipeline.pkl')

# print the model
print(model)

# creating a confusion matrix
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

actual = y_test
predictions = model.predict(X_test)

cm = confusion_matrix(actual, predictions, normalize=None)

unique_targets = sorted(list(y_test))

x = Y = sorted(list(df1[label].unique()))
# showing it on the browser
fig = ff.create_annotated_heatmap(cm, x, Y)

fig.update_layout(title_text="<b>Confusion matrix</b>",
                  yaxis=dict(categoryorder="category descending"))

fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted label",
                        xref="paper",
                        yref="paper"))

fig.add_annotation(dict(font=dict(color="black", size=14),
                        x=-0.15,
                        y=0.5,
                        showarrow=False,
                        text="Actual label",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# We need margins so the titles fit
fig.update_layout(margin=dict(t=80, r=20, l=120, b=50))
fig['data'][0]['showscale'] = True
fig.show()

# testing the model
print()
print('Training Set: %d, Test Set: %d \n' % (X_train.shape[0], X_test.shape[0]))

df1_predictions = model.predict(X_test)
print('Predicted labels: ', df1_predictions[:15])
print('Actual labels   : ', y_test[:15])

from sklearn.metrics import classification_report

# classification report
print()
print(classification_report(y_test, df1_predictions))

# overall of the accuracy, precision, recall from the test predictions
print("Overall Accuracy:", accuracy_score(y_test, df1_predictions))
print("Overall Precision:", precision_score(y_test, df1_predictions, average='macro'))
print("Overall Recall:", recall_score(y_test, df1_predictions, average='macro'))
# putting it on the confusion matrix
mcm = confusion_matrix(y_test, df1_predictions)

print()
print(mcm)

df1_classes = ['best_Sales', 'worst_Sales']  # just wrote this but it can be change.

import numpy as np

# plotting the confusion matrix
plt.imshow(mcm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(df1_classes))
plt.xticks(tick_marks, df1_classes, rotation=45)
plt.yticks(tick_marks, df1_classes)
plt.title("Confusion matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()

