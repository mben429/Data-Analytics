# importing k means from scikit
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale

# We need to cluster the data into 2 seperate groups. (0 or 1) -> Genuine banknote vs forged banknote
# The Data was extracted from images that were taken from genuine and forged banknote-like specimens

# Loading in data, Has 5 columns, id, V1, V2, V3, V4
# V1 = entropy
#      measurement of disorder, uncertainty, lower means easier to predict
# V2 = skewness
#      Symetric/Normal? Means no skewness, is data being pulled in any direction?
#      Positive towards right, negative towards left
# V3 = Variance
#      The average of the squared differences from the Mean, (HIGH VARIANCE = Overfit)
# V4 = kurtosis of the transformation of the banknote (measure of the tailedness) (Kind of how pointy)
#      Positive means very pointy, negative flat, low kurtosis, little outliers

raw_bn_data = pd.read_csv("BanknoteData.csv")
# If there are missing values, fill with average
"""
Method: 

- Lets compute the means of each colunm

"""

means = raw_bn_data.mean()
mean_entropy = means[1]
mean_skewness = means[2]
mean_variance = means[3]
mean_kurtosis = means[4]

new_means = [mean_entropy, mean_skewness, mean_variance, mean_kurtosis]



def db_scan_k_means(raw_bn_data):
    dbscan_raw_data = []
    processed_data = raw_bn_data.drop(["ID", "V1", "V4"], axis=1)

    DBSCAN_algorithm = cluster.DBSCAN(eps=0.4, min_samples=7, metric="euclidean")
    dbscan_raw_data = DBSCAN_algorithm.fit_predict(processed_data).tolist()

    # Assign outlier values as forged
    list_of_outlier_pos = []
    count = 0
    for i in range(len(dbscan_raw_data)):
        if dbscan_raw_data[i] == -1:
            count += 1
            list_of_outlier_pos += [i]
    print("Number of outliers = ", count)
    print("Percentage of outliers = ", count/len(dbscan_raw_data))

    X_clustered = []

    k_means = KMeans(n_clusters=2, max_iter=100, n_init=10, init="k-means++", random_state=100)
    X_clustered = k_means.fit_predict(processed_data).tolist()

    # This assigns outliers to the forged class
    for i in range(len(X_clustered)):
        if i in list_of_outlier_pos:
            X_clustered[i] = 1

    return X_clustered


def get_percentages_to_csv(list_pred):
    percentage_0 = list_pred.count(0) / len(list_pred)
    percentage_1 = list_pred.count(1) / len(list_pred)

    print("Resulting list: ", list_pred)
    print("Percentage Forged: ", percentage_0)
    print("Percentage Genuine: ", percentage_1)

    returning_df = raw_bn_data.drop(["V1", "V2", "V3", "V4"], axis=1)
    returning_df["Class"] = list_pred
    returning_df.to_csv("tst_kaggle_A5.csv", index=False)


#get_percentages_to_csv(method_1(raw_bn_data))
get_percentages_to_csv(db_scan_k_means(raw_bn_data))
