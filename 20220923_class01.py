# 선형회귀의 과적합으로 인해 개선을 시켜주는 필요성
# 라소(L1), 릿지(L2)
# 예측모델 = w1 * x1 + w2 * x2 .... + b
# 라소(L1) w1,w2.... 가중치를 0에 가깝게 만든다.
# 릿지(L2)  w1,w2.... 가중치를 0에 가깝게 만든다.
# 라소, 릿지의 차이점은
#  라소는 실제로 0을 만든다. - 변수가 선택
#  릿지는 실제로 0을 만들지는 않는다. - 모든 변수가 존재.
# 이를 통해서 과대적합이 어느정도 해소된다.


import mglearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
import pandas as pd
import numpy as np

# 한글
import matplotlib
from matplotlib import font_manager, rc
font_loc = "C:/Windows/Fonts/malgunbd.ttf"
font_name = font_manager.FontProperties(fname=font_loc).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False

# %matplotlib inline

### 데이터 셋 준비
boston = load_boston()  # 데이터 셋 불러오기
print(type(boston.target), type(boston.data))
print(boston.target.shape, boston.data.shape)

df_boston = pd.DataFrame(boston.data,columns=boston.feature_names)
df_boston['target'] = pd.Series(boston.target)

pd.set_option('display.max_columns', None)
print( df_boston.head() )

# 보스턴 데이터 셋을 불러오는 또 다른 3가지 방법
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# from sklearn.datasets import fetch_california_housing
# housing = fetch_california_housing()

# from sklearn.datasets import fetch_openml
# housing = fetch_openml(name="house_prices", as_frame=True)

# 사전준비
# 입력/출력
# 0~1사이로 만들기
# 변수 확장

# 입력/출력 선택
X = df_boston.loc[:, 'CRIM':'LSTAT']
y = df_boston['target']

# 0~1사이로 만들기
nor_X = MinMaxScaler().fit_transform(X)  # 정규화

# 변수 확장
ex_X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(nor_X)

print("전의 상태 : ", X.shape, np.min(X), np.max(X))
print("적용된 상태 : ", ex_X.shape, np.min(ex_X), np.max(ex_X))

# 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(ex_X, y, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 회귀 모델
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
print("모델 평가 결과(결정계수)-학습 : ", model.score(X_train, y_train))
print("모델 평가 결과(결정계수)-테스트용 : ", model.score(X_test, y_test))
print("(회귀모델) 최종 남겨진 변수 개수 : ", format(np.sum(model.coef_ != 0)))
print()
# (추가) 5:5~9:1까지의 평가결과를 확인하고 비교해보기(댓글- 올려보기)
# (실습) 데이터 나누어 회귀 모델 적용시켜서 이의 결과를 반환하는 함수화 시켜보기(댓글 - 올려보기)

# 01. 선형회귀 모델을 만든다.
# 02. 라쏘회귀 모델을 만든다. - w의 값을 0을 만드는 친구가 생긴다.
# from sklearn.linear_model import Lasso, Ridge
model_L = Lasso(alpha=0.01)
model_L.fit(X_train, y_train)
print("모델 평가 결과(결정계수)-학습 : ", model_L.score(X_train, y_train))
print("모델 평가 결과(결정계수)-테스트용 : ", model_L.score(X_test, y_test))
print("(라쏘모델) 최종 남겨진 변수 개수 : ", format(np.sum(model_L.coef_ != 0)))
print()

# 03. 릿지회귀 모델을 만든다. - w의 값을 0을 만드는 친구가 생기지 않는다.
model_R = Ridge()
model_R.fit(X_train, y_train)
print("모델 평가 결과(결정계수)-학습 : ", model_R.score(X_train, y_train))
print("모델 평가 결과(결정계수)-테스트용 : ", model_R.score(X_test, y_test))
print("(릿지모델) 최종 남겨진 변수 개수 : ", format(np.sum(model_R.coef_ != 0)))
print()
# 04. 모델을 비교해본다.