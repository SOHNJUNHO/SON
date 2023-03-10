import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("train.csv") # Product Category

data = data.drop(['id'], axis = 1)

mapping_dict = {"Class_1": 1,
                "Class_2": 2,
                "Class_3": 3,
                "Class_4": 4,
                "Class_5": 5,
                "Class_6": 6,
                "Class_7": 7,
                "Class_8": 8,
                "Class_9": 9}
                
after_mapping_target = data['target'].apply(lambda x: mapping_dict[x])

feature_columns = list(data.columns.difference(['target'])) # target을 제외한 모든 행
X = data[feature_columns] # 설명변수
y = after_mapping_target # 타겟변수
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42) # 학습데이터와 평가데이터의 비율을 8:2 로 분할| 
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # 데이터 개수 확인

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
random_forest_model1 = RandomForestClassifier(n_estimators = 20, # 20번 추정
                                             max_depth = 5, # 트리 최대 깊이 5
                                             random_state = 42) # 시드값 고정
model1 = random_forest_model1.fit(train_x, train_y) # 학습 진행
predict1 = model1.predict(test_x) # 평가 데이터 예측
print("Accuracy: %.2f" % (accuracy_score(test_y, predict1) * 100), "%") # 정확도 % 계산

print(np.__version__, pd.__version__)