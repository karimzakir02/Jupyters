import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import warnings
warnings.filterwarnings("ignore")


houses = pd.read_csv("datasets/houses_to_rent_v2.csv")
houses.head()


houses.shape


# Getting the numerical columns
num_cols = houses.select_dtypes(include=np.number).columns.tolist()
num_cols


houses["floor"].unique()


houses["floor"] = houses["floor"].replace({"-": "0"}).astype(int)


num_cols = houses.select_dtypes(include=np.number).columns.tolist()
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle("Data Distribution of Numerical Columns")
for index, col in enumerate(num_cols):
    if index < 5:  
        sns.histplot(ax=axs[0][index], data=houses, x=col)
    else:
        sns.histplot(ax=axs[1][index-5], data=houses, x=col)
plt.show()


fig, axs = plt.subplots(1, 5, figsize=(20, 8))
fig.suptitle("Data Distribution of Numerical Columns")
area_plot = sns.histplot(ax=axs[0], data=houses, x="area")
area_plot.set(xlim=(0, 1000))
hoa_plot = sns.histplot(ax=axs[1], data=houses, x="hoa (R$)")
hoa_plot.set(xlim=(0, 6000))
pt_plot = sns.histplot(ax=axs[2], data=houses, x="property tax (R$)")
pt_plot.set(xlim=(0, 2000))
total_plot = sns.histplot(ax=axs[3], data=houses, x="total (R$)")
total_plot.set(xlim=(0, 30000))
floor_plot = sns.histplot(ax=axs[4], data=houses, x="floor")
floor_plot.set(xlim=(0, 40))
plt.show()


potential_features = ["city", "area", "rooms", "bathroom", "parking spaces", "floor", "animal", "furniture"]
target = ["rent amount (R$)"]

X = houses[potential_features]
y = houses[target]


cat_cols = houses.select_dtypes(exclude=np.number).columns.tolist()


fig, axs = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("Value Count for Each Categorical Column")
for index, col in enumerate(cat_cols):
    sns.countplot(ax=axs[index], data=houses, x=col)
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)


X_train_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
X_test_missing = [col for col in X_test.columns if X_test[col].isnull().any()]
y_train_missing = [col for col in y_train.columns if y_train[col].isnull().any()]
y_test_missing = [col for col in y_test.columns if y_test[col].isnull().any()]


print("Columns with missing values in X_train:", X_train_missing)
print("Columns with missing values in X_test:", X_test_missing)
print("Columns with missing values in y_train:", y_train_missing)
print("Columns with missing values in y_test:", y_test_missing)


# Checking for potentially missing values in X_train
X_train_area = X_train["area"] <= 0
X_train_rooms = X_train["rooms"] <= 0
X_train_bathroom = X_train["bathroom"] <= 0
X_train[(X_train_area) | (X_train_rooms) | (X_train_bathroom)]


# Checking for potentially missing values in X_test 
X_test_area = X_test["area"] <= 0
X_test_rooms = X_test["rooms"] <= 0
X_test_bathroom = X_test["bathroom"] <= 0
X_test[(X_test_area) | (X_test_rooms) | (X_test_bathroom)]


X_train[X_train["parking spaces"] < 0]


X_test[X_test["parking spaces"] < 0]


X_train[X_train["parking spaces"] == 0].describe()


X_test[X_test["parking spaces"] == 0].describe()


no_parking_train = X_train[X_train["parking spaces"] == 0]
fig, axs = plt.subplots(1, 2)
area_boxplot = sns.boxplot(y="area", data=no_parking_train, ax=axs[0])
area_boxplot.set(ylim=(0, 850))
room_boxplot = sns.boxplot(y="rooms", data=no_parking_train, ax=axs[1])
plt.show()


k = 1.5
q1 = no_parking_train["area"].quantile(0.25)
q3 = no_parking_train["area"].quantile(0.75)
iqr = q3 - q1
threshold = q3 + k*iqr
above_threshold_train = no_parking_train[no_parking_train["area"] > threshold]


len(above_threshold_train) / len(X_train) * 100


no_parking_test = X_test[X_test["parking spaces"] == 0]
above_threshold_test = no_parking_test[no_parking_test["area"] > threshold]


len(above_threshold_test) / len(X_test) * 100


X_train[["area", "rooms", "parking spaces"]].corr()


X_train.drop(above_threshold_train.index, inplace=True)
X_test.drop(above_threshold_test.index, inplace=True)


y_train.drop(above_threshold_train.index, inplace=True)
y_test.drop(above_threshold_test.index, inplace=True)


y_train[y_train["rent amount (R$)"] < 1]


y_test[y_test["rent amount (R$)"] < 1]


houses.select_dtypes(exclude=np.number).columns.tolist()


X_train["animal"] = X_train["animal"] == "acept"
X_train["animal"] = X_train["animal"].astype(int)
X_test["animal"] = X_test["animal"] == "acept"
X_test["animal"] = X_test["animal"].astype(int)


X_train["furniture"] = X_train["furniture"] == "furnished"
X_train["furniture"] = X_train["furniture"].astype(int)
X_test["furniture"] = X_test["furniture"] == "furnished"
X_test["furniture"] = X_test["furniture"].astype(int)


len(X_train["city"].unique()) == len(X_test["city"].unique())


oh_encoder = OneHotEncoder(sparse=False)
ohe_city_train = pd.DataFrame(oh_encoder.fit_transform(X_train[["city"]]))
ohe_city_train.index = X_train.index
ohe_city_test = pd.DataFrame(oh_encoder.transform(X_test[["city"]]))
ohe_city_test.index = X_test.index


X_train.drop("city", inplace=True, axis=1)
X_test.drop("city", inplace=True, axis=1)


X_train = X_train.merge(ohe_city_train, right_index=True, left_index=True)
X_test = X_test.merge(ohe_city_test, right_index=True, left_index=True)


training_dataset = X_train.merge(y_train, left_index=True, right_index=True)
inv_cov = np.linalg.matrix_power(training_dataset.cov(), -1)
mean = np.mean(training_dataset)
training_dataset["mahalanobis_distance"] = np.NaN


for index, row in training_dataset.iterrows():
    values = row[0:13]
    training_dataset.loc[index, "mahalanobis_distance"] = mahalanobis(values, mean, inv_cov)**2


mahalanobis_threshold = chi2.ppf(0.99, training_dataset.shape[1] - 1)
mahalanobis_outliers = training_dataset[training_dataset["mahalanobis_distance"] > mahalanobis_threshold]
len(mahalanobis_outliers)


mahalanobis_threshold


mahalanobis_outliers.sort_values("mahalanobis_distance", ascending=False)


training_dataset[training_dataset["mahalanobis_distance"] >= 100]


training_dataset[(training_dataset["area"] > 650) & (training_dataset["area"] < 750)].drop(2182).describe()


to_remove = [2562, 5915, 2182, 2397]


training_dataset[(training_dataset["mahalanobis_distance"] < 100) & (training_dataset["mahalanobis_distance"] >= 60)].sort_values(by="mahalanobis_distance", ascending=False)


to_remove.extend([1946, 5445, 10619, 2619, 6947, 4813, 1810, 1639])


training_dataset[(training_dataset["mahalanobis_distance"] < 60) & (training_dataset["mahalanobis_distance"] >= 50)].sort_values(by="mahalanobis_distance", ascending=False)


to_remove.extend([1946, 10125, 9857, 5525, 3066, 9012])


training_dataset[(training_dataset["mahalanobis_distance"] < 50) & (training_dataset["mahalanobis_distance"] >= 40)].sort_values(by="mahalanobis_distance", ascending=False)


to_remove.extend([4829, 5998, 9442, 9479, 1130, 9788, 9464, 2845, 3853, 6384, 6851, 801, 9822, 10376, 4088, 1571, 1780, 6423, 2398, 3853, 4634, 2867, 20, 8380, 6014])


training_dataset[(training_dataset["mahalanobis_distance"] < 40) & (training_dataset["mahalanobis_distance"] >= 35)].sort_values(by="mahalanobis_distance", ascending=False)


to_remove.extend([4497, 3348, 9995, 510, 6954, 1764, 5293, 4027, 317, 9568, 8943, 1838, 8334, 2106, 7597, 7226, 1138, 9856, 8399])


training_dataset[(training_dataset["mahalanobis_distance"] < 35) & (training_dataset["mahalanobis_distance"] >= 31)].sort_values(by="mahalanobis_distance", ascending=False)


to_remove.extend([2275, 6482, 3485, 863, 4594, 1697, 1568, 10558, 1159, 9775, 10416, 10082, 6996, 2642, 680, 8067, 8587, 8080, 9663, 911, 5685, 6219, 7675, 6506, 816, 2627])


training_dataset[(training_dataset["mahalanobis_distance"] < 31) & (training_dataset["mahalanobis_distance"] >= mahalanobis_threshold)].sort_values(by="mahalanobis_distance", ascending=False)


to_remove.extend([676, 6673, 595, 1719, 3092, 8327, 2859, 6185, 2624, 6823, 4719, 4170, 9833, 1491, 770, 8678, 4025, 9948, 5763, 10530, 10392, 5611, 7192, 1743, 1877, 4681, 5288, 607, 6202])


X_train.drop(to_remove, inplace=True)
y_train.drop(to_remove, inplace=True)


testing_dataset = X_test.merge(y_test, left_index=True, right_index=True)
testing_dataset["mahalanobis_distance"] = np.NaN


for index, row in testing_dataset.iterrows():
    values = row[0:13]
    testing_dataset.loc[index, "mahalanobis_distance"] = mahalanobis(values, mean, inv_cov)**2


len(testing_dataset[testing_dataset["mahalanobis_distance"] > mahalanobis_threshold])


testing_dataset[testing_dataset["mahalanobis_distance"] > mahalanobis_threshold].sort_values("mahalanobis_distance", ascending=False)


to_remove_test = [8874, 1676, 10472, 7826, 8312, 6504, 4517, 6753, 628, 2315, 6777, 2998, 1976, 8362, 4993, 7742, 2338, 8628, 2576, 8966, 10587, 1426, 5660, 7748, 1528, 7835, 6101, 4224]


X_test.drop(to_remove_test, inplace=True)
y_test.drop(to_remove_test, inplace=True)


training_dataset = X_train.merge(y_train, left_index=True, right_index=True)


sns.scatterplot(data=training_dataset, x="area", y="rent amount (R$)")


training_dataset[training_dataset["area"] >= 2000]


training_dataset.drop(8790, inplace=True)
X_train.drop(8790, inplace=True)
y_train.drop(8790, inplace=True)


sns.scatterplot(data=training_dataset, x="area", y="rent amount (R$)")


training_dataset[["area", "rent amount (R$)"]].corr()


# Consider plotting at different axis, so that you can properly see the range of the area differences
# This should also be done for other analysis
counter = 100
counters = []
rent_diffs = []
area_diffs = []
while counter < 400:
    counters.append(counter)
    training_segment = training_dataset[(training_dataset["area"] < counter+10) & (training_dataset["area"] >= counter)]
    rooms_2 = training_segment[training_segment["rooms"] == 2]
    rooms_4 = training_segment[training_segment["rooms"] == 4]
    rooms_2_area = rooms_2[["area", "rent amount (R$)"]].mean().loc["area"]
    rooms_4_area = rooms_4[["area", "rent amount (R$)"]].mean().loc["area"]
    area_diffs.append(rooms_2_area - rooms_4_area)
    rooms_2_rent = rooms_2[["area", "rent amount (R$)"]].mean().loc["rent amount (R$)"]
    rooms_4_rent = rooms_4[["area", "rent amount (R$)"]].mean().loc["rent amount (R$)"]
    rent_diffs.append(rooms_2_rent - rooms_4_rent)
    counter += 10


to_plot_rent = pd.DataFrame({"counters": counters, "diffs": rent_diffs}, columns=["counters", "diffs"])
to_plot_rent["label"] = "rent differences"
to_plot_area = pd.DataFrame({"counters": counters, "diffs": area_diffs}, columns=["counters", "diffs"])
to_plot_area["label"] = "area differences"
to_plot = pd.concat([to_plot_rent, to_plot_area], ignore_index=True)


sns.lineplot(x="counters", y="diffs", hue="label", data=to_plot)


counter = 100
counters = []
rent_diffs = []
area_diffs = []
room_diffs = []
while counter < 250:
    counters.append(counter)
    training_segment = training_dataset[(training_dataset["area"] < counter+10) & (training_dataset["area"] >= counter)]
    bathroom_1 = training_segment[training_segment["bathroom"] == 1]
    bathroom_3 = training_segment[training_segment["bathroom"] == 3]
    bathroom_1_area = bathroom_1[["area", "rent amount (R$)"]].mean().loc["area"]
    bathroom_3_area = bathroom_3[["area", "rent amount (R$)"]].mean().loc["area"]
    area_diffs.append(bathroom_1_area - bathroom_3_area)
    bathroom_1_rooms = bathroom_1[["area", "rent amount (R$)", "rooms"]].mean().loc["rooms"]
    bathroom_3_rooms = bathroom_3[["area", "rent amount (R$)", "rooms"]].mean().loc["rooms"]
    room_diffs.append(bathroom_1_rooms - bathroom_3_rooms)
    bathroom_1_rent = bathroom_1[["area", "rent amount (R$)"]].mean().loc["rent amount (R$)"]
    bathroom_3_rent = bathroom_3[["area", "rent amount (R$)"]].mean().loc["rent amount (R$)"]
    rent_diffs.append(bathroom_1_rent - bathroom_3_rent)
    counter += 10


to_plot_rent = pd.DataFrame({"counters": counters, "diffs": rent_diffs}, columns=["counters", "diffs"])
to_plot_rent["label"] = "rent differences"
to_plot_area = pd.DataFrame({"counters": counters, "diffs": area_diffs}, columns=["counters", "diffs"])
to_plot_area["label"] = "area differences"
to_plot_rooms = pd.DataFrame({"counters": counters, "diffs": room_diffs}, columns=["counters", "diffs"])
to_plot_rooms["label"] = "room differences"
to_plot = pd.concat([to_plot_rent, to_plot_area, to_plot_rooms], ignore_index=True)


sns.lineplot(x="counters", y="diffs", hue="label", data=to_plot)


counter = 100
counters = []
rent_diffs = []
area_diffs = []
room_diffs = []
while counter < 300:
    counters.append(counter)
    training_segment = training_dataset[(training_dataset["area"] < counter+10) & (training_dataset["area"] >= counter)]
    bathroom_1 = training_segment[training_segment["parking spaces"] == 2]
    bathroom_3 = training_segment[training_segment["parking spaces"] == 4]
    bathroom_1_area = bathroom_1[["area", "rent amount (R$)"]].mean().loc["area"]
    bathroom_3_area = bathroom_3[["area", "rent amount (R$)"]].mean().loc["area"]
    area_diffs.append(bathroom_1_area - bathroom_3_area)
    bathroom_1_rooms = bathroom_1[["area", "rent amount (R$)", "rooms"]].mean().loc["rooms"]
    bathroom_3_rooms = bathroom_3[["area", "rent amount (R$)", "rooms"]].mean().loc["rooms"]
    room_diffs.append(bathroom_1_rooms - bathroom_3_rooms)
    bathroom_1_rent = bathroom_1[["area", "rent amount (R$)"]].mean().loc["rent amount (R$)"]
    bathroom_3_rent = bathroom_3[["area", "rent amount (R$)"]].mean().loc["rent amount (R$)"]
    rent_diffs.append(bathroom_1_rent - bathroom_3_rent)
    counter += 10


to_plot_rent = pd.DataFrame({"counters": counters, "diffs": rent_diffs}, columns=["counters", "diffs"])
to_plot_rent["label"] = "rent differences"
to_plot_area = pd.DataFrame({"counters": counters, "diffs": area_diffs}, columns=["counters", "diffs"])
to_plot_area["label"] = "area differences"
to_plot_rooms = pd.DataFrame({"counters": counters, "diffs": room_diffs}, columns=["counters", "diffs"])
to_plot_rooms["label"] = "room differences"
to_plot = pd.concat([to_plot_rent, to_plot_area, to_plot_rooms], ignore_index=True)


sns.lineplot(x="counters", y="diffs", hue="label", data=to_plot)


training_dataset[["floor", "rent amount (R$)"]].corr()


training_dataset[training_dataset["floor"] == 0][["area", "rent amount (R$)"]].mean()


training_dataset[training_dataset["floor"] >= 0][["area", "rent amount (R$)"]].mean()


fig, axs = plt.subplots(1, 2, figsize=(12,4))
sns.barplot(data=training_dataset[["animal", "rent amount (R$)"]].groupby("animal").mean().reset_index(), y="rent amount (R$)", x="animal", ax=axs[0])
axs[0].set_title("Average rent for apartments which allow/disallow animals")
sns.countplot(data=training_dataset, x="animal", ax=axs[1])
axs[1].set_title("No. animal-friendly/unfriendly apartments in the dataset")
plt.show()


training_dataset[["area", "rooms", "rent amount (R$)", "animal"]].groupby("animal").mean()


animal_group = training_dataset[["area", "rooms", "rent amount (R$)", "animal"]].groupby("animal").get_group(1)
animal_group.sort_values("area", ascending=False, inplace=True)
area_mean = animal_group["area"].mean()
while area_mean > 105:
    animal_group = animal_group.iloc[1:]
    area_mean = animal_group["area"].mean()


animal_group["area"].mean()


animal_group["rooms"].mean()


animal_group["rent amount (R$)"].mean()


mean_animal = animal_group["rent amount (R$)"].mean()
no_animal_mean = 3298.690987
animal_test_statistic = mean_animal - no_animal_mean


no_animal_group = training_dataset[["area", "rooms", "rent amount (R$)", "animal"]].groupby("animal").get_group(0)
total_animal = pd.concat([animal_group, no_animal_group])


# Make sure that this is consistently calculated in the right order
np.random.seed(22)
sim_data = []
repetitions = 1000
for i in range(repetitions):
    animal_copy = total_animal.copy()
    animal_copy["shuffled_labels"] = pd.DataFrame(animal_copy["animal"].values, index=np.random.permutation(animal_copy.index))
    means = animal_copy.groupby("shuffled_labels").mean()["rent amount (R$)"].values.tolist()
    diff = means[0] - means[1]
    sim_data.append(diff)


sns.histplot(x=sim_data)


total = 0
for sim in sim_data:
    if sim > abs(animal_test_statistic) or sim < animal_test_statistic:
        total += 1
total / 1000 * 100


fig, axs = plt.subplots(1, 2, figsize=(12,4))
sns.barplot(data=training_dataset[["furniture", "rent amount (R$)"]].groupby("furniture").mean().reset_index(), y="rent amount (R$)", x="furniture", ax=axs[0])
axs[0].set_title("Average rent for furnished/unfurnished apartments")
sns.countplot(data=training_dataset, x="furniture", ax=axs[1])
axs[1].set_title("No. furnished/unfurnished apartments in the dataset")
plt.show()


training_dataset[["area", "rooms", "rent amount (R$)", "furniture"]].groupby("furniture").mean()


furniture_means = furniture_X_train.groupby("furniture").mean()["rent amount (R$)"].values.tolist()
test_statistic = furniture_means[0] - furniture_means[1]
test_statistic


np.random.seed(33)
sim_data = []
repetitions = 1000
for i in range(repetitions):
    furniture_copy = training_dataset.copy()
    furniture_copy["shuffled_labels"] = pd.DataFrame(furniture_copy["furniture"].values, index=np.random.permutation(furniture_copy.index))
    means = furniture_copy.groupby("shuffled_labels").mean()["rent amount (R$)"].values.tolist()
    diff = means[0] - means[1]
    sim_data.append(diff)


sns.histplot(x=sim_data)


total = 0
for sim in sim_data:
    if sim > abs(test_statistic) or sim < test_statistic:
        total += 1
total / 1000 * 100


training_dataset = training_dataset.merge(houses["city"], how="inner", left_index=True, right_index=True)


sns.countplot(data=training_dataset, x="city")


fig, axs = plt.subplots(1, 2, figsize=(12,5))
mean_cities = training_dataset.groupby("city")[["rent amount (R$)", "area"]].mean()
sns.barplot(data=mean_cities, x=mean_cities.index, y="rent amount (R$)", ax=axs[0])
sns.barplot(data=mean_cities, x=mean_cities.index, y="area", ax=axs[1])


X_train.drop(["animal", "floor"], axis=1, inplace=True)
X_test.drop(["animal", "floor"], axis=1, inplace=True)


init_model = XGBRegressor(n_estimators=10)
init_model.fit(X_train, y_train)


init_predictions = init_model.predict(X_test)
mean_absolute_error(init_predictions, y_test)


km_clf = KMeans(n_clusters=3)
km_clf.fit(X_train.iloc[:, 0:12])


X_train["km_labels"] = pd.DataFrame(km_clf.labels_, index=X_train.index)
y_train["km_labels"] = pd.DataFrame(km_clf.labels_, index=y_train.index)


test_labels = km_clf.predict(X_test)
X_test["km_labels"] = pd.DataFrame(test_labels, index=X_test.index)
y_test["km_labels"] = pd.DataFrame(test_labels, index=y_test.index)


km_groups_X_train = X_train.groupby("km_labels")
km_groups_y_train = y_train.groupby("km_labels")


km_groups_X_test = X_test.groupby("km_labels")
km_groups_y_test = y_test.groupby("km_labels")


km_error_measurements = {}
for name, group in km_groups_X_train:
    km_group_model = XGBRegressor(n_estimators=10)
    group_X_train = group.iloc[:, 0:12]
    group_y_train = km_groups_y_train.get_group(name)[["rent amount (R$)"]]
    km_group_model.fit(group_X_train, group_y_train)
    group_X_test = km_groups_X_test.get_group(name).iloc[:, 0:12]
    group_y_test = km_groups_y_test.get_group(name)[["rent amount (R$)"]]
    group_predictions = km_group_model.predict(group_X_test)
    km_error_measurements[name] = mean_absolute_error(group_predictions, group_y_test)


km_error_measurements


sns.countplot(data=X_test, x="km_labels")


ms_clf = MeanShift()
ms_clf.fit(X_train.iloc[:, 0:12])


X_train["ms_labels"] = pd.DataFrame(ms_clf.labels_, index=X_train.index)
y_train["ms_labels"] = pd.DataFrame(ms_clf.labels_, index=y_train.index)


ms_groups_X_train = X_train.groupby("ms_labels")
ms_groups_y_train = y_train.groupby("ms_labels")


sns.countplot(data=X_train, x="ms_labels")


print(f"There are {len(ms_groups_X_train.get_group(0))} in group 0")
print(f"There are {len(ms_groups_X_train.get_group(1))} in group 1")
print(f"There are {len(ms_groups_X_train.get_group(2))} in group 1")
print(f"There are {len(ms_groups_X_train.get_group(3))} in group 3")
print(f"There are {len(ms_groups_X_train.get_group(4))} in group 4")


y_test["initial predictions"] = pd.DataFrame(init_predictions, index=y_test.index)
y_test["initial error"] = abs(y_test["rent amount (R$)"] - y_test["initial predictions"])


km_groups_y_test = y_test.groupby("km_labels")


y_test.groupby("km_labels").mean()


846.394984 / 2602.001970 * 100


2443.138294 / 6624.610245 * 100


3539.218286 / 10444.926829 * 100
