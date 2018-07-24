# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:03:32 2018

@author: 이혁주
"""
#Minsuk Heo, Youtube 채널 참고

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #setting seaborn default for plots

# importing Classifier Modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Cross Validation(K-fold)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits = 10, shuffle = True, random_state=0)



train = pd.read_csv("C:\python\Train.csv")
test = pd.read_csv("C:\python\Test.csv")

#print(train.head())

#print(train.shape)

#print(train.info())

#print(train.isnull().sum())

'''시각화 단계'''
#bar-chart
'''
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked=True, figsize=(10,5))

bar_chart('Sex')
bar_chart('Pclass')
'''

train_test_data = [train, test]

for dataset in train_test_data:
    dataset["Title"] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    

#print(train['Title'].value_counts())
    
#title mapping
title_mapping = {"Mr":0, "Miss":1, "Mrs":2,
                 "Master":3, "Dr":3, "Rev":3, "Col":3,
                 "Major":3, "Mlle":3, "Countess": 3,
                 "Ms":3, "Lady":3, "Jonkheer":3, "Don":3, "Dona":3,
                 "Mme":3 , "Capt":3, "Sir":3}

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    


#sex mapping
sex_mapping = {"male":0, "female":1}    # mapping 중괄호

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


#bar_chart('Sex')

#print(train.head(3))    #위에서 부터 행 몇 개만 보기

#Name Column 삭제
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
#print(train.head(3))

#Age Column 결측치 처리. 방법 fillna, 혹은 dropna
#dropna: 
#train.dropna(thresh=2) NAN이 2개 이상인 것 삭제.

#groupby함수.
#train.groupby('Company').count() 혹은 max()

a = train.groupby('Title').median() #단계확인
a2= train.groupby('Title')['Age'].transform("median")   #단계 이해 ****

#print(a['Age']) #단계확인


train["Age"].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)

#transform 변화.
#fillna 값 채울 때 원리 파악!

#빙잉?
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
#loc indexing은 label 숫자고려
    
'''value_counts()함수
print(train['Embarked'].value_counts())
print(train[train['Pclass']==1]['Embarked'].value_counts())
print(train[train['Pclass']==2]['Embarked'].value_counts())
print(train[train['Pclass']==3]['Embarked'].value_counts())
'''
#위와 같은 결과를 시각화를 통해 전처리 방향 결정 가능

#Embarked 결측치 처리 - numeric value 변환

for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('S')
    
Embarked_mapping = {'S':0, 'C':1, 'Q':2}

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(Embarked_mapping)

train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'), inplace = True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace = True)

#시각화 이후 판단하에 categorical 변수로 변환
#기법 빙 어쩌구

for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3
#loc 함수 이해
#[] 이 안의 조건에서, ,이후 의 값을 할당

#print(train.Cabin.value_counts())

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
# 첫번째 글자만 고려

#cabin mapping
cabin_mapping = {'A':0, 'B':0.4, 'C':0.8, 'D': 1.2, 'E':1.6, 'F':2, 'G':2.4, "T":2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

#cabin missing value
train["Cabin"].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace = True)
test['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace = True)

#FamilySize (sibsp + parch)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

features_drop = ['Ticket', 'SibSp', 'Parch']

train = train.drop(features_drop, axis = 1)
test = test.drop(features_drop, axis = 1)
train = train.drop(['PassengerId'], axis = 1)

train_data = train.drop('Survived', axis = 1)
target = train['Survived']




'''Classifier 사용'''
#K-Neighbors
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
#print(score)

#print(round(np.mean(score)*100,2)) # 평균 accuracy 반올림 값

#SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring = scoring)


'''SVM선택해서 제출까지'''
'''
clf = SVC()
clf.fit(train_data, target)

test_data = test.drop('PassengerId', axis=1).copy()
prediction = clf.predict(test_data)


submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': prediction
        })

submission.to_csv('Submission.csv', index=False)

submission = pd.read_csv('Submission.csv')
'''







