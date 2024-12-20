import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드
file_path = './data/7.lpg_leakage.csv'
data = pd.read_csv(file_path, names=['Alcohol', 'CH4', 'CO', 'H2', 'LPG', 'Propane', 'Smoke', 'Temp', 'LPG_Leakage'])

# 2. 데이터 전처리
# 결측값 처리 및 숫자 변환
print("데이터의 첫 5줄:\n", data.head())
data = data.apply(pd.to_numeric, errors='coerce').dropna()

# 독립 변수와 종속 변수 분리
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 데이터 스케일링
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 로지스틱 회귀 모델
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# 평가
print("\n[Logistic Regression]")
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))
print("Classification Report:\n", classification_report(y_test, y_pred_logistic))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logistic))

# 4. 랜덤 포레스트 모델
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 평가
print("\n[Random Forest]")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# 5. 시각화
# 혼동 행렬 시각화
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
plt.matshow(conf_matrix_rf, cmap='Blues', fignum=1)
plt.colorbar()
plt.title('Confusion Matrix: Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('./results/confusion_matrix_rf.png')
print("혼동 행렬 저장 완료: ./results/confusion_matrix_rf.png")

# 상관계수 히트맵 (텍스트로 출력)
correlation_matrix = data.corr()
print("\n상관계수:\n", correlation_matrix)

# 피처 중요도 텍스트 출력 (랜덤 포레스트)
importances = rf_model.feature_importances_
features = X.columns
sorted_indices = np.argsort(importances)[::-1]
print("\nFeature Importances (Random Forest):")
for idx in sorted_indices:
    print(f"{features[idx]}: {importances[idx]:.4f}")