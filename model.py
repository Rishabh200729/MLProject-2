# %% [markdown]
# # Dragon Real Estate - Price Predictor

# %%
import pandas as pd

# %%
housing = pd.read_csv("data.csv")

# %%
%matplotlib inline

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# ## Train Test Split

# %%
# import numpy as np

# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size =int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_set_size]

# %%
# train_set , test_set = split_train_test(housing, 0.2)

# %%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

# %%
print(f"Rows in train set : {len(train_set)} \nRows in test set: {len(test_set)} ")

# %%
from sklearn.model_selection import StratifiedShuffleSplit 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index , test_index in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# %%
housing = strat_train_set.copy()

# %% [markdown]
# ## Looking for correlations

# %%
# from pandas.plotting import scatter_matrix
# attributes = ['MEDV','RM', 'ZN','LSTAT']
# scatter_matrix(housing[attributes], figsize=(12,8))

# %%
# housing.plot(kind="scatter", x ='RM', y='MEDV', alpha=0.8)

# %% [markdown]
# ## Trying out Attribute combinations 

# %%
housing['TAXRM'] = housing['TAX']/housing['RM']
housing.head()

# %%
# corr_matrix = housing.corr()
# corr_matrix["MEDV"].sort_values(ascending=False)

# %%
housing = strat_train_set.drop('MEDV', axis = 1)
housing_labels = strat_train_set['MEDV'].copy()

# %% [markdown]
# ## Missing Attributes
# 

# %%
# to take care of missing attribuets , u have 3 options
    # TO GET RID OF THE MISSING DATA POINTS
    # GET RID OF THE WHOLE ATTRIBUTE
    # SET THE VALUE TO SOME VALUE (0, MEAN, MEDIAN)
# from sklearn.impute  import SimpleImputer
# imputer = SimpleImputer(strategy = "median")
# imputer.fit(housing)


# %%
# imputer.statistics_

# %%
# X = imputer.transform(housing)

# %%
# housing_tr = pd.DataFrame(X, columns=housing.columns)

# %%
# housing_tr.describe()

# %% [markdown]
# ## Scikit-learn Design

# %% [markdown]
# #### Primarily three types of objects 1.Estimators 2. Transformers 3. Predictors
# #### Estimators -  Ex - Imputer
# #### Transformers - transforms input and gives output
# #### Predictors - Linear Regression , K nearest neighbour 

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scalar', StandardScaler())
])

# %% [markdown]
# ## Feature Scaling

# %%
housing_num = my_pipeline.fit_transform(housing)

# %%
housing_num
housing_num.shape

# %% [markdown]
# #### Select and train model

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num, housing_labels)

# %%
some_data = housing.iloc[:5]

# %%
some_labels = housing_labels.iloc[:5]

# %%
prepared_data = my_pipeline.transform(some_data)

# %%
model.predict(prepared_data)

# %%
list(some_labels)

# %% [markdown]
# ### Evaluating the model

# %% [markdown]
# 

# %%
import numpy as np
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# %%
rmse

# %% [markdown]
# ### Cross Validation

# %%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num, housing_labels, scoring = "neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# %%
rmse_scores

# %%
def print_scores(scores):
    print("scores :", scores)
    print("mean:", scores.mean())
    print("Standard Deviation : ", scores.std())

# %%
print_scores(rmse_scores)
# %%
