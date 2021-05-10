import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.api as sm
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


houses = pd.read_csv("houses_to_rent_v2.csv")
houses.head()


# Defining the features/target and splitting the data
features = ["city", "area", "rooms", "bathroom", "parking spaces", "floor", "animal", "furniture"]
target = ["total (R$)"]
X = houses[features]
y = houses[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)


# Preparing categorical data with One Hot Encoding
cat_cols = ["city", "animal", "furniture"]
city_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

oh_city_train = pd.DataFrame(city_encoder.fit_transform(X_train[cat_cols]))
oh_city_test = pd.DataFrame(city_encoder.transform(X_test[cat_cols]))

oh_city_train.index = X_train.index
oh_city_test.index = X_test.index

X_train.drop(cat_cols, axis=1, inplace=True)
X_test.drop(cat_cols, axis=1, inplace=True)

X_train = X_train.merge(oh_city_train, left_index=True, right_index=True)
X_test = X_test.merge(oh_city_test, left_index=True, right_index=True)


# Floor has some non-numerical values ('-'), these will be replaced by 0, since I am assuming there are no floors, and the data is not missing
X_train.replace({"-": 0}, inplace=True)
X_test.replace({"-": 0}, inplace=True)
X_train["floor"] = X_train["floor"].astype(int)
X_test["floor"] = X_test["floor"].astype(int)
# Apart from this, there is no other potentially missing data in the dataset


default_model = LinearRegression()
default_model.fit(X_train, y_train)
default_predictions = default_model.predict(X_test)
baseline_mae = mean_absolute_error(default_predictions, y_test)
baseline_mae


houses[target].mean()


applicable_cols = ["area", "rooms", "bathroom", "floor", "total (R$)"]
k = 1.5
num_out_train = X_train.merge(y_train, left_index=True, right_index=True)
to_remove = {}
for col in applicable_cols:
    quantiles = num_out_train[[col]].quantile([0.25, 0.75])
    q1 = quantiles.iloc[0, 0]
    q3 = quantiles.iloc[1, 0]
    iqr = q3 - q1
    lower_bound = q1 - k*iqr
    upper_bound = q3 + k*iqr
    records_to_remove = num_out_train[(num_out_train[col] > upper_bound) | (num_out_train[col] < lower_bound)].index
    to_remove[col] = set(records_to_remove)


combined = set()
for indexes in to_remove.values():
    combined.update(indexes)
print("Removed:", len(combined))
len(combined) / len(X_train) * 100


num_out_X_train = X_train.drop(list(combined), axis=0)
num_out_y_train = y_train.drop(list(combined), axis=0)
num_out_regress = LinearRegression()
num_out_regress.fit(num_out_X_train, num_out_y_train)
num_out_predictions = num_out_regress.predict(X_test)
num_out_mae = mean_absolute_error(num_out_predictions ,y_test)
num_out_mae


baseline_mae - num_out_mae


num_out_dropped = num_out_train.loc[list(combined)]
custom_groups = {}
for key, values in to_remove.items():
    removed_obsv = num_out_dropped.loc[values]
    custom_groups[key] = removed_obsv


summary_table = pd.DataFrame(["mean", "count", "min", "max", "std"], columns=["metric"])
summary_table.set_index("metric", inplace=True)
for key in custom_groups.keys():
    summary_table[key] = np.NaN


for key, group in custom_groups.items():
    summary_table.loc["mean", key] = group[key].mean()
    summary_table.loc["count", key] = group[key].count()
    summary_table.loc["min", key] = group[key].min()
    summary_table.loc["max", key] = group[key].max()
    summary_table.loc["std", key] = group[key].std()
summary_table


# Check that this is actually how you're supposed to do it
sm_model = sm.OLS(y_train, X_train).fit()
influence = sm_model.get_influence()
cooks = influence.cooks_distance[0]
cooks_df = pd.DataFrame(cooks, columns=["cooks"])
cooks_df.index = X_train.index
X_train_cooks = X_train.merge(cooks_df, left_index=True, right_index=True)


cooks_threshold = 4/len(X_train_cooks)
X_train_cooks_filtered = X_train_cooks[X_train_cooks["cooks"] < cooks_threshold]
y_train_cooks_filtered = y_train.loc[X_train_cooks_filtered.index]


# Saving the removed observations for later analysis
cooks_removed_X = X_train_cooks[X_train_cooks["cooks"] >= cooks_threshold]
cooks_removed_y = y_train.loc[cooks_removed_X.index]


X_train_cooks_filtered.drop(["cooks"], axis=1, inplace=True)


cooks_model = LinearRegression()
cooks_model.fit(X_train_cooks_filtered, y_train_cooks_filtered)
cooks_predictions = cooks_model.predict(X_test)
cooks_mae = mean_absolute_error(cooks_predictions, y_test)
cooks_mae


baseline_mae - cooks_mae


len(X_train_cooks) - len(X_train_cooks_filtered)


cooks_removed_X.merge(cooks_removed_y, left_index=True, right_index=True)


mahalanobis_train = X_train.merge(y_train, left_index=True, right_index=True)
inv_covmat = np.linalg.pinv(mahalanobis_train.cov())
# I have used pseudo-inverse, since the np library throws a Singular Matrix error when using the standard inverse
mean = np.mean(mahalanobis_train)
mahalanobis_train["mahalanobis_distance"] = np.NaN


for index, values in mahalanobis_train.iterrows():
    values = values[0:15]
    mahalanobis_train.loc[index, "mahalanobis_distance"] = mahalanobis(values, mean, inv_covmat)


mahalanobis_threshold = chi2.ppf(0.95, mahalanobis_train.shape[1] - 1)
mahalanobis_filtered = mahalanobis_train[mahalanobis_train["mahalanobis_distance"] < mahalanobis_threshold]
mahalanobis_removed = mahalanobis_train[mahalanobis_train["mahalanobis_distance"] >= mahalanobis_threshold]


mahalanobis_X_test = mahalanobis_filtered.iloc[:,0:14]
mahalanobis_y_test = mahalanobis_filtered.iloc[:, 14]
mahalanobis_model = LinearRegression()
mahalanobis_model.fit(mahalanobis_X_test, mahalanobis_y_test)
mahalanobis_predictions = mahalanobis_model.predict(X_test)
mahalanobis_mae = mean_absolute_error(mahalanobis_predictions, y_test)
mahalanobis_mae


baseline_mae - mahalanobis_mae


mahalanobis_mae - cooks_mae


len(mahalanobis_removed)


mahalanobis_removed


mahalanobis_threshold_80 = chi2.ppf(0.8, mahalanobis_train.shape[1] - 1)
mahalanobis_train_80_removed = mahalanobis_train[mahalanobis_train["mahalanobis_distance"] >= mahalanobis_threshold_80]
mahalanobis_train_80_removed


mahalanobis_train_80 = mahalanobis_train[mahalanobis_train["mahalanobis_distance"] < mahalanobis_threshold_80]
mahalanobis_X_test_80 = mahalanobis_train_80.iloc[:,0:14]
mahalanobis_y_test_80 = mahalanobis_train_80.iloc[:, 14]
mahalanobis_model_80 = LinearRegression()
mahalanobis_model_80.fit(mahalanobis_X_test_80, mahalanobis_y_test_80)
mahalanobis_predictions_80 = mahalanobis_model_80.predict(X_test)
mahalanobis_mae_80 = mean_absolute_error(mahalanobis_predictions_80, y_test)
mahalanobis_mae_80


dbscan_train = X_train.merge(y_train, left_index=True, right_index=True)
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(dbscan_train)


unique = []
outlier_count = 0
for value in dbscan.labels_:
    if value not in unique:
        unique.append(value)
    if value == -1:
        outlier_count += 1
outlier_count


outlier_count / len(X_train) * 100


eps_list = [i for i in np.arange(0.5, 200.5, 1)]
mae_list = []
removed_list = []
for eps in eps_list:
    dbscan = DBSCAN(eps=eps)
    dbscan.fit(dbscan_train)
    labels_df = pd.DataFrame(dbscan.labels_, columns=["labels"])
    labels_df.index = dbscan_train.index
    current_copy = dbscan_train.copy()
    current_copy = current_copy.merge(labels_df, right_index=True, left_index=True)
    removed = len(current_copy[current_copy["labels"] == -1])
    removed_list.append(removed)
    current_copy = current_copy[current_copy["labels"] >= 0]
    dbscan_X = current_copy.iloc[:, :14]
    dbscan_y = current_copy.iloc[:, 14]
    dbscan_regress = LinearRegression()
    dbscan_regress.fit(dbscan_X, dbscan_y)
    dbscan_predictions = dbscan_regress.predict(X_test)
    dbscan_mae = mean_absolute_error(dbscan_predictions, y_test)
    mae_list.append(dbscan_mae)    


fig, ax = plt.subplots(1, 2, figsize=(12,8))
fig.suptitle("DBSCAN Testing")
ax[0].set_title("Linear Regression MAE")
ax[1].set_title("Amount of Observations Removed")
sns.lineplot(x=eps_list, y=mae_list, ax=ax[0])
sns.scatterplot(x=eps_list, y=removed_list, ax=ax[1])


eps = 50.5
dbscan_50 = DBSCAN(eps=eps)
dbscan_50.fit(dbscan_train)
labels_df_50 = pd.DataFrame(dbscan_50.labels_, columns=["labels"])
labels_df_50.index = dbscan_train.index
outliers_index = labels_df_50[labels_df_50["labels"] == -1].index
outliers_dbscan = dbscan_train.loc[outliers_index]
len(outliers_dbscan)


outliers_dbscan.describe()


outliers_dbscan[outliers_dbscan["area"] == 30]


outliers_dbscan.sample(15)
