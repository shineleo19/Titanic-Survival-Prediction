import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import zipfile
import os

zip_path = r"d:\ai project\archive (1).zip"
extract_path = "/mnt/data/iris_dataset"
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Files extracted to:", extract_path)

for file in os.listdir(extract_path):
    if file.endswith(".csv"):
        data_path = os.path.join(extract_path, file)
        break

df = pd.read_csv(data_path)
print("\n📄 Dataset Preview:")
print(df.head())

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n🌳 Decision Tree Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n⚙️ Logistic Regression Results:")
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
