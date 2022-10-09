from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
y = boston.target

# 01. 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 02. 모델 생성
model = LinearRegression()

# 03. 모델 학습
model.fit(X_train, y_train)

# 04. 모델 예측
pred = model.predict(X_test)
print(pred)

# 05. MSE 구해보기