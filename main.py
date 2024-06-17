import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

data = pd.read_csv("D:\PRODIGY_ML_01\train.csv")

missing_percentage = data.isnull().mean() * 100
missing_columns = missing_percentage[missing_percentage > 0]

fill_values = {
    'LotFrontage': 69,
    'Alley': 'No Alley',
    'MasVnrType': 'None',
    'FireplaceQu': 'No Fireplace',
    'PoolQC': 'No Pool',
    'Fence': 'No Fence',
    'MiscFeature': 'No Feature'
}

group_fill_values = {
    'No Garage': ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond'],
    'No Basement': ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtExposure', 'BsmtFinType2']
}

for col, value in fill_values.items():
    data[col] = data[col].fillna(value)

for value, cols in group_fill_values.items():
    data[cols] = data[cols].fillna(value)

data = data.drop(columns='Id', axis=1)

cols_with_null_vals = data.columns[data.isnull().sum() > 0]

data = data.dropna(subset=['MasVnrArea', 'Electrical'])

ord_vars = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
data[ord_vars] = data[ord_vars].astype('object')

categorical_vars = data.select_dtypes(include='object').columns
numerical_vars = data.select_dtypes(exclude='object').columns


print('Total categorical features =', categorical_vars.shape[0])
print('Total numerical features =', numerical_vars.shape[0])


for var in numerical_vars:
    if var != 'SalePrice':  
        sns.scatterplot(data=data, x=var, y='SalePrice')
        plt.title(f'Scatter plot of {var} vs SalePrice')
        plt.xlabel(var)
        plt.ylabel('SalePrice')
        # plt.show()

numerical_vars = data.select_dtypes(exclude='object').columns


plt.figure(figsize=(12, 10))
sns.heatmap(data[numerical_vars].corr(), cmap='YlGnBu', annot=True, fmt='.2f', annot_kws={'size': 10})

bin = [var for var in data.columns if len(data[var].unique()) == 2]

data['Street'] = data['Street'].map({'Pave': 1, 'Grvl': 0})
data['Utilities'] = data['Utilities'].map({'AllPub': 1, 'NoSeWa': 0})
data['CentralAir'] = data['CentralAir'].map({'Y': 1, 'N': 0})

categorical_vars = data.select_dtypes(include='object').columns

le = LabelEncoder()

for var in categorical_vars:
    if var == 'GarageYrBlt':
        data[var] = data[var].apply(lambda x: str(x)[:-2] if x != 'No Garage' else x)
    else:
        data[var] = le.fit_transform(data[var])

numerical_vars = data.select_dtypes(exclude='object').columns

le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

train_data, test_data = train_test_split(data, train_size=0.7, test_size=0.3, random_state=42)  

X_train = train_data.drop(columns='SalePrice', axis=1)
y_train = train_data['SalePrice']
X_test = test_data.drop(columns='SalePrice', axis=1)
y_test = test_data['SalePrice']

scalerX = MinMaxScaler()
X_train_scaled = pd.DataFrame(scalerX.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scalerX.transform(X_test), columns=X_test.columns)

scalerY = MinMaxScaler()
y_train_scaled = scalerY.fit_transform(np.array(y_train).reshape(-1, 1))
y_test_scaled = scalerY.transform(np.array(y_test).reshape(-1, 1))

X = data.drop(columns='SalePrice')
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

alphas = []
for i in range(-4, 3):
    for j in range(1, 10):
        val = j * (10**i)
        alphas.append(round(val, abs(i)))

folds = KFold(n_splits=5, shuffle=True, random_state=100)

param_grid = {'alpha': alphas}

cv_ridge = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_absolute_error', cv=folds, return_train_score=True)
cv_ridge.fit(X_train, y_train)

best_alpha = cv_ridge.best_params_['alpha']

rm = Ridge(alpha=best_alpha).fit(X_train, y_train)

y_train_pred = rm.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

y_test_pred = rm.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)

residuals = y_train - y_train_pred


lasso_cv = GridSearchCV(Lasso(), param_grid, scoring='neg_mean_absolute_error', cv=folds, return_train_score=True)
lasso_cv.fit(X_train, y_train)

best_alpha = lasso_cv.best_params_['alpha']

lm = Lasso(alpha=best_alpha).fit(X_train, y_train)

y_train_pred = lm.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

y_test_pred = lm.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)


residuals = y_train - y_train_pred


test_data = pd.read_csv("D:\PRODIGY_ML_01\test.csv")

ord_vars = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']
test_data[ord_vars] = test_data[ord_vars].astype('object')

fill_modes = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
for col in fill_modes:
    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])

fill_medians = ['LotFrontage', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
for col in fill_medians:
    test_data[col] = test_data[col].fillna(test_data[col].median())

fill_means = ['MasVnrArea']
for col in fill_means:
    test_data[col] = test_data[col].fillna(test_data[col].mean())

fill_values = {
    'Alley': 'No alley access', 'MasVnrType': 'None', 'BsmtQual': 'No Basement', 'BsmtCond': 'No Basement',
    'BsmtExposure': 'No Basement', 'BsmtFinType1': 'No Basement', 'BsmtFinType2': 'No Basement',
    'FireplaceQu': 'No Fireplace', 'GarageType': 'No Garage', 'GarageYrBlt': test_data['GarageYrBlt'].mode()[0],
    'GarageFinish': 'No Garage', 'GarageQual': 'No Garage', 'GarageCond': 'No Garage', 'PoolQC': 'No Pool',
    'Fence': 'No Fence', 'MiscFeature': 'No Feature'
}
for col, val in fill_values.items():
    test_data[col] = test_data[col].fillna(val)

test_data = test_data.drop(columns='Id')

binary_mappings = {
    'Street': {'Pave': 1, 'Grvl': 0},
    'Utilities': {'AllPub': 1, 'NoSeWa': 0},
    'CentralAir': {'Y': 1, 'N': 0}
}
for col, mapping in binary_mappings.items():
    test_data[col] = test_data[col].map(mapping)

categorical_vars = test_data.select_dtypes(include='object').columns
le = LabelEncoder()
for var in categorical_vars:
    test_data[var] = le.fit_transform(test_data[var])

scalerX = MinMaxScaler()
test_data = scalerX.fit_transform(test_data)


X_train = train_data.drop(columns='SalePrice')
y_train = train_data['SalePrice']
y_train = np.array(y_train).reshape(-1, 1)

X_train = scalerX.fit_transform(X_train)
scalerY = MinMaxScaler()
y_train = scalerY.fit_transform(y_train)

rm_final = Ridge(alpha=cv_ridge.best_params_['alpha']).fit(X_train, y_train)

predictions = rm_final.predict(test_data)
predictions = scalerY.inverse_transform(predictions.reshape(-1, 1))
predictions = predictions.reshape(-1, )

submission_df = pd.DataFrame({
    'Id': pd.read_csv("D:\PRODIGY_ML_01\test.csv")['Id'],
    'SalePrice': predictions
})

submission_df.to_csv('submission_py.csv', index=False)
