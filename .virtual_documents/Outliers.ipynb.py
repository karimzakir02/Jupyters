# Importing the required libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.cluster import DBSCAN
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# Loading the dataset and saving its length
houses = pd.read_csv("houses_to_rent_v2.csv")
original_length = len(houses)
houses.head()


# The floor column has some non-numerical values ('-'), these will be replaced by 0, since I am assuming that '-' indicates a single-story house and 
# does not indicate missing data
houses.replace({"-": 0}, inplace=True)
houses["floor"] = houses["floor"].astype(int)
# Apart from this, there is no other potentially missing data in the dataset
houses = houses.select_dtypes(include=np.number)


# Getting the observations which are identified as outliers by the Numeric Outlier Technique
numeric_cols = houses.select_dtypes(include=np.number).columns.tolist()
k = 1.5
to_remove = {}
for col in numeric_cols:
    quantiles = houses[[col]].quantile([0.25, 0.75])
    q1 = quantiles.iloc[0, 0]
    q3 = quantiles.iloc[1, 0]
    iqr = q3 - q1
    lower_bound = q1 - k*iqr
    upper_bound = q3 + k*iqr
    num_out_outliers_index = houses[(houses[col] > upper_bound) | (houses[col] < lower_bound)].index
    to_remove[col] = set(num_out_outliers_index)


# Determining the amount of observations removed
combined = set()
for indexes in to_remove.values():
    combined.update(indexes)
print("Removed:", len(combined))
len(combined) / original_length * 100


# Getting a slice of the data containing only outliers for each column
num_out_outliers = houses.loc[list(combined)]
custom_groups = {}
for key, values in to_remove.items():
    removed_obsv = houses.loc[values]
    custom_groups[key] = removed_obsv


# Preparing to create a summary table
summary_table = pd.DataFrame(["mean", "count", "min", "max", "std"], columns=["metric"])
summary_table.set_index("metric", inplace=True)
for key in custom_groups.keys():
    summary_table[key] = np.NaN


# Populating the summary table
for key, group in custom_groups.items():
    summary_table.loc["mean", key] = group[key].mean()
    summary_table.loc["count", key] = group[key].count()
    summary_table.loc["min", key] = group[key].min()
    summary_table.loc["max", key] = group[key].max()
    summary_table.loc["std", key] = group[key].std()
summary_table


# Preparing the quantiles and the IQR for each column
quantiles_dict = {}
for col in numeric_cols:
    quantiles = houses[[col]].quantile([0.25, 0.75])
    q1 = quantiles.iloc[0, 0]
    q3 = quantiles.iloc[1, 0]
    iqr = q3 - q1
    quantiles_dict[col] = [q1, q3, iqr]


# Finding the IQR mutlitplier so that less than 1% of the dataset is removed
currently_removed = len(combined)
removed_count = [len(combined)]
removed_indexes = {}
k_lst = [k]
one_percent = int(0.01 * original_length)
while currently_removed > one_percent:
    k += 0.5
    k_lst.append(k)
    for col in numeric_cols:
        q1 = quantiles_dict[col][0]
        q3 = quantiles_dict[col][1]
        iqr = quantiles_dict[col][2]
        lower_bound = q1 - k*iqr
        upper_bound = q3 + k*iqr
        num_out_outliers_index = houses[(houses[col] > upper_bound) | (houses[col] < lower_bound)].index
        to_remove[col] = set(num_out_outliers_index)
    combined = set()
    for indexes in to_remove.values():
        combined.update(indexes)
    removed_indexes[k] = list(combined)
    currently_removed = len(combined)
    removed_count.append(currently_removed)
k


# Plotting the number of points identified as outliers against the IQR mutlitpliers tested
plt.title("Number Points Identified as Outliers with Different IQR Multipliers")
plt.xlabel("IQR Multiplier")
plt.ylabel("Number of Outliers")
sns.scatterplot(x=k_lst, y=removed_count)


# Creating a summary table for the new batch of outliers with the IQR mutltipler being 8
summary_table = pd.DataFrame(["mean", "count", "min", "max", "std"], columns=["metric"])
summary_table.set_index("metric", inplace=True)
k = 8
for col in numeric_cols:
    quantiles = houses[[col]].quantile([0.25, 0.75])
    q1 = quantiles.iloc[0, 0]
    q3 = quantiles.iloc[1, 0]
    iqr = q3 - q1
    lower_bound = q1 - k*iqr
    upper_bound = q3 + k*iqr
    num_out_outliers = houses[(houses[col] > upper_bound) | (houses[col] < lower_bound)]
    summary_table[col] = np.NaN
    summary_table.loc["mean", col] = num_out_outliers[col].mean()
    summary_table.loc["count", col] = num_out_outliers[col].count()
    summary_table.loc["min", col] = num_out_outliers[col].min()
    summary_table.loc["max", col] = num_out_outliers[col].max()
    summary_table.loc["std", col] = num_out_outliers[col].std()
summary_table


# Finding the inverse of the covariance matrix and the center
inv_cov = np.linalg.matrix_power(houses.cov(), -1)
mean = np.mean(houses)
mahalanobis_houses = houses.copy().select_dtypes(include=np.number)
mahalanobis_houses["mahalanobis_distance"] = np.NaN


# Adding the distance to the copy of the dataset
for index, values in mahalanobis_houses.iterrows():
    values = values[0:10]
    mahalanobis_houses.loc[index, "mahalanobis_distance"] = mahalanobis(values, mean, inv_cov)**2


# Finding the cut-off and identifying outliers
mahalanobis_threshold = chi2.ppf(0.99, mahalanobis_houses.shape[1] - 1)
mahalanobis_removed = mahalanobis_houses[mahalanobis_houses["mahalanobis_distance"] > mahalanobis_threshold].index
len(mahalanobis_removed)


# Determing the proportion of the dataset that the outliers take up
len(mahalanobis_removed) / original_length * 100


# Getting a sample of the outliers
houses.loc[mahalanobis_removed].sample(n=20, random_state=10)


houses.loc[mahalanobis_removed].sample(n=20, random_state=20)


# Preparing our features and targets
features = houses.columns[0:9]
target = houses.columns[9]
X = houses[features]
y = houses[target]


# Training a model and getting Cook's Distance for each point
X = sm.tools.tools.add_constant(X)
sm_model = sm.regression.linear_model.OLS(y, X).fit()
influence = sm_model.get_influence()
influence_list = influence.cooks_distance[0]
influence_df = pd.DataFrame(influence_list, columns=["influence"])
influence_df.index = houses.index
cooks_df = houses.merge(influence_df, left_index=True, right_index=True)


cooks_threshold = 4/original_length
cooks_outliers = cooks_df[cooks_df["influence"] > cooks_threshold]
len(cooks_outliers)


len(cooks_outliers) / original_length * 100


cooks_outliers.sort_values(by=["influence"])


filtered_houses = houses[(houses["area"] >= 89) & (houses["area"] <= 99)]
filtered_houses = filtered_houses[(filtered_houses["rooms"] >= 2) & (filtered_houses["rooms"] <= 4)]
filtered_houses = filtered_houses[(filtered_houses["bathroom"] >= 1) & (filtered_houses["bathroom"] <= 5)]
filtered_houses = filtered_houses[(filtered_houses["parking spaces"] >= 1) & (filtered_houses["parking spaces"] <= 3)]
filtered_houses = filtered_houses[(filtered_houses["hoa (R$)"] >= 700) & (filtered_houses["hoa (R$)"] <= 800)]
filtered_houses = filtered_houses[(filtered_houses["rent amount (R$)"] >= 2333) & (filtered_houses["rent amount (R$)"] <= 2533)]
filtered_houses = filtered_houses[(filtered_houses["property tax (R$)"] >= 0) & (filtered_houses["property tax (R$)"] <= 100)]
filtered_houses = filtered_houses[(filtered_houses["fire insurance (R$)"] >= 21) & (filtered_houses["fire insurance (R$)"] <= 41)]
filtered_houses.drop(labels=[4994], axis=0)


cooks_df.loc[6175]["influence"]


dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(houses)


unique = []
outlier_count = 0
for value in dbscan.labels_:
    if value not in unique:
        unique.append(value)
    if value == -1:
        outlier_count += 1
outlier_count


outlier_count / original_length * 100


currently_removed = outlier_count
eps = 1
min_samples = 5
removed_lst = []
eps_lst = []
while currently_removed > one_percent:
    eps_lst.append(eps)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(houses)
    labels_df = pd.DataFrame(dbscan.labels_, columns=["labels"])
    labels_df.index = houses.index
    removed_df = labels_df[labels_df["labels"] == -1]
    removed_indexes = removed_df.index
    currently_removed = len(removed_indexes)
    removed_lst.append(currently_removed)
    eps += 1000


# fig, ax = plt.subplots(1, 2, figsize=(12,8))
# fig.suptitle("DBSCAN Testing")
# ax[0].set_title("Linear Regression MAE")
# ax[1].set_title("Amount of Observations Removed")
sns.lineplot(x=eps_lst, y=removed_lst)
# sns.scatterplot(x=eps_list, y=removed_list, ax=ax[1])





houses.loc[removed_indexes]
