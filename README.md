# Predicting-Student-Grades-ML-
import pandas as pd
import copy
import numpy as np
from sklearn.preprocessing import LabelEncoder

df= pd.read_csv('student-mat.csv')

df['school'].value_counts()
df['school']

df.info()

df


df['school'] = df['school'].map( {'GP': 1, 'MS': 0} ).astype(int)
df['sex'] = df['sex'].map( {'F':1, 'M':0} ).astype(int)
df['address'] = df['address'].map( {'U': 1, 'R': 0} ).astype(int)
df['famsize'] = df['famsize'].map( {'GT3': 1, 'LE3': 0} ).astype(int)
df['Pstatus'] = df['Pstatus'].map( {'T': 1, 'A': 0} ).astype(int)
df['schoolsup'] = df['schoolsup'].map( {'no': 1, 'yes': 0} ).astype(int)
df['famsup'] = df['famsup'].map( {'no': 0, 'yes': 1} ).astype(int)
df['paid'] = df['paid'].map( {'no': 1, 'yes': 0} ).astype(int)
df['activities'] = df['activities'].map( {'no': 0, 'yes': 1} ).astype(int)
df['nursery'] = df['nursery'].map( {'no': 0, 'yes': 1} ).astype(int)
df['higher'] = df['higher'].map( {'no': 0, 'yes': 1} ).astype(int)
df['internet'] = df['internet'].map( {'no': 0, 'yes': 1} ).astype(int)
df['romantic'] = df['romantic'].map( {'no': 1, 'yes': 0} ).astype(int)

# One hot encoding
emb=pd.get_dummies(df['Mjob'],columns='Mjob',prefix='Mjob')
df=pd.concat([df, emb], axis=1)
df.drop(['Mjob'],axis=1,inplace= True)

emb=pd.get_dummies(df['Fjob'],columns='Fjob',prefix='Fjob')
df=pd.concat([df, emb], axis=1)
df.drop(['Fjob'],axis=1,inplace= True)

emb=pd.get_dummies(df['reason'],columns='reason',prefix='reason')
df=pd.concat([df, emb], axis=1)
df.drop(['reason'],axis=1,inplace= True)

emb=pd.get_dummies(df['guardian'],columns='guardian',prefix='guardian')
df=pd.concat([df, emb], axis=1)
df.drop(['guardian'],axis=1,inplace= True)


df.info()

df.iloc[:,44:]

#df['score']=0

#for i in range(395):
    #if df.iloc[i,26]=='A':
        #df.iloc[i,44]=4
    #if df.iloc[i,26]=='B':
        #df.iloc[i,44]=3
    #if df.iloc[i,26]=='C':
        #df.iloc[i,44]=2
    #if df.iloc[i,26]=='F':
        #df.iloc[i,44]=1

#df['score'].nunique()

df.info()

#df.drop(['grade'],axis=1,inplace=True)

X = df.drop('grade', axis = 1)
y = df['grade']


df.info()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)

X_test_df = pd.DataFrame(X_test, columns= X.columns)

#1
X_test_df.describe()['age']

#Softmax Regression

#from sklearn.linear_model import LinearRegression
#lreg= LinearRegression()

#lreg.fit(X_train,y_train)
#print('Train Score : ' , lreg.score(X_train,y_train))
#print('Test Score : ' , lreg.score(X_test,y_test))

#Logistic regression for Pass or Fail
y = y.map( {'A': 'Pass', 'B': 'Pass','C': 'Pass','F': 'Fail'} ).astype(str)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)

logisreg = LogisticRegression()

logisreg.fit(X_train,y_train)

print('Train Score : ' , logisreg.score(X_train,y_train))
print('Test Score : ' , logisreg.score(X_test,y_test))

# SoftmaxRegression

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=1,max_iter=1000,random_state=0)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict_proba(X_train)

softmax_reg.score(X_test,y_test)

svm = SVC(kernel='rbf', gamma=0.1, C=1,random_state=0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
data = {'Actual':y_test.to_numpy(),'Predict':y_pred}
Test = pd.DataFrame(data)
Test_B = Test[Test['Actual']=='B']
len(Test_B[(Test_B['Actual'] == 'B') & (Test_B['Predict'] == 'B')])

y_pred

# Decision Trees Classifier

%matplotlib notebook
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

dtree = DecisionTreeClassifier(max_depth=2,random_state=0,criterion='entropy')

dtree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(dtree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dtree.score(X_test, y_test)))

import matplotlib.pyplot as plt
import numpy as np

dtree.fit(X_train, y_train)
tree.plot_tree(dtree,precision=4)

# SVM in RBF

from sklearn.svm import SVC

clf3 = SVC(kernel='rbf', gamma=0.1, C=1,random_state=0)
clf3.fit(X_train,y_train)
y_pred= clf3.predict(X_test)
y_test

Test.loc[Test['Actual']=='B']

data = {'Actual':y_test.to_numpy(),'Predict':y_pred}
Test = pd.DataFrame(data)
Test

# Nutrition Dataset

nut_df = pd.read_csv('Nutritional Data for Fast Foods.csv')
nut_df

nut_catdf = nut_df.select_dtypes(include=['object']).copy()
nut_catdf

print(nut_catdf.isnull().values.sum())

nut_catdf.at[2,'Item'] = None
nut_catdf

nut_catdf.isnull()

nut_catdf['Fast Food Restaurant'].value_counts()

nut_df.loc[nut_df['Fast Food Restaurant']=="Hardee's" ]

nut_df.groupby('Type').describe()['Trans Fat (g)']

nut_df_grouped = nut_df.groupby("Type")
nut_imputed = nut_df_grouped.transform(lambda grp: grp.fillna(grp.mean()))
nut_df['Trans Fat (g)']=nut_imputed['Trans Fat (g)'] 
nut_df
nut_df.loc[nut_df['Fast Food Restaurant']=="Hardee's" ]


nut_df['Trans Fat (g)'].isnull()
nut_df.loc[nut_df['Fast Food Restaurant']=="Hardee's" ]

nut_df.groupby("Type").describe()['Trans Fat (g)']

nut_df.describe()['Trans Fat (g)']

nut_df_groupedfat = nut_df.groupby("Type")
nut_imputed = nut_df_grouped.transform(lambda grp: grp.fillna(grp.mean()))
nut_df['Total Fat (g)']=nut_imputed['Total Fat (g)'] 
nut_df.describe()['Total Fat (g)']

nut_df_groupedsodium = nut_df.groupby("Type")
nut_imputed = nut_df_grouped.transform(lambda grp: grp.fillna(grp.mean()))
nut_df['Sodium (mg)']=nut_imputed['Sodium (mg)'] 
nut_df.describe()['Sodium (mg)']

nut_df.isnull().sum()

nut_df_groupedprotein = nut_df.groupby("Type")
nut_imputed = nut_df_grouped.transform(lambda grp: grp.fillna(grp.mean()))
nut_df['Protein (g)']=nut_imputed['Protein (g)'] 
nut_df.describe()['Protein (g)']

nut_df.isnull().sum()


nut_df['Fast Food Restaurant'].value_counts()

#DummyDataforprocessing

list_copyobj = nut_copy['Fast Food Restaurant','Type']
list_copyobj

nut_copy = nut_df
df_nutcopyencoded = pd.get_dummies(data=nut_copy, columns=['Type','Fast Food Restaurant'])
df_nutcopyencoded.info()
nut_df = df_nutcopyencoded.drop(['Item'],axis=1)
nut_df.info


nut_df.rename(columns={'Unnamed: 0':'Index'}, inplace=True)
#nut_df=nut_df.set_index('Unnamed: 0')
nut_df

X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_org)
X_test = scaler.transform(X_test_org)
X_train_org
X_test


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pandas import Series, DataFrame
%matplotlib inline
import matplotlib.pyplot as plt
from  sklearn.linear_model import Lasso

target = nut_df['Calories']

X_nut= nut_df.drop('Calories',axis=1)
y_nut=nut_df.Calories

X_test_df= pd.DataFrame(X_test,columns=X.columns)


X_train

from sklearn.linear_model import LinearRegression

lreg = LinearRegression()
lreg.fit(X_train, y_train)
print(lreg.score(X_train, y_train))
print(lreg.score(X_test, y_test))

predictors = X_train_org.columns

coef = Series(lreg.coef_,predictors).sort_values()

coef.plot(kind='bar', title='Modal Coefficients')

from  sklearn.linear_model import Ridge

#x_range = [0.01, 0.1, 1, 10, 100,35,15,0.5]
#train_score_list = []
#test_score_list = []

#for alpha in x_range: 
ridge = Ridge(35)
ridge.fit(X_train,y_train)
    #train_score_list.append(ridge.score(X_train,y_train))
    #test_score_list.append(ridge.score(X_test, y_test))


ridge.coef_

print(train_score_list)
print(test_score_list)

from sklearn.linear_model import Lasso
#x_range = [0.01, 0.1, 1, 10, 100,0.35,35,15,0.15]
#train_score_list = []
#test_score_list = []

#for alpha in x_range: 
lasso = Lasso(4)
lasso.fit(X_train,y_train)
    #train_score_list.append(lasso.score(X_train,y_train))
    #test_score_list.append(lasso.score(X_test, y_test))

lasso.coef_

lasso.score(X_train,y_train)


lasso.score(X_test,y_test)

plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')
plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')
plt.xscale('log')
plt.legend(loc = 3)
plt.xlabel(r'$\alpha$')

##Polynomial Regression

from  sklearn.preprocessing  import PolynomialFeatures
train_score_list_poly = []
test_score_list_poly = []

#for n in range(2,3):
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
lreg.fit(X_train_poly, y_train)
train_score_list_poly.append(lreg.score(X_train_poly, y_train))
test_score_list_poly.append(lreg.score(X_test_poly, y_test))

from  sklearn.preprocessing  import PolynomialFeatures
poly = PolynomialFeatures(2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

print(train_score_list_poly)
print(test_score_list_poly)


print(lreg.score(X_train_poly, y_train))
print(lreg.score(X_test_poly, y_test))

X_train_poly.shape

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

tree= DecisionTreeRegressor().fit(X_train,y_train)
