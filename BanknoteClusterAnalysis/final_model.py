from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

data = pd.read_csv("BanknoteData.csv")

# Preprocessing

# Stage 1: Missing data check
print("Number of missing values in Data:", data.isnull().sum().sum())

# Stage 2: Remove ID column
new_df = data.drop(["ID"], axis=1)

# Stage 4: Principal Component Analysis
# We want to reduce dimensionality as much as possible, right now we are sitting at 4 dimensions, we wish to reduce to
# maybe 2 or 3 to keep the accuracy whilst removing the complexity

# a) PCA is affected by scale, so we should standardize the data first
std = StandardScaler().fit_transform(new_df)

# b) Lets try normalizing
nrm = Normalizer().fit_transform(new_df)

# Now onto the model, we are going to try KMeans, where n-Clusters = 2.
def k_means_nrm():
    # Almost default K_means
    km = KMeans(n_clusters=2)
    prediction = km.fit_predict(nrm)
    list_prediction = list(prediction)
    centroids = km.cluster_centers_
    plt.scatter(nrm[:, 0], nrm[:, 1], c=prediction, s=50, cmap="viridis")
    plt.title("KMeans with Normalization")
    plt.show()
    return flip_results(list_prediction)

def k_means_std():
    # Almost default K_means
    km = KMeans(n_clusters=2)
    prediction = list(km.fit_predict(std))
    plt.scatter(std[:, 0], std[:, 1], c=prediction, s=50, cmap="viridis")
    plt.title("KMeans with Standardization")
    plt.show()
    return prediction

def std_pca_kmeans(components):
    pca = PCA(n_components=components)
    p_components = pca.fit_transform(std)
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='orange')
    plt.title("PCA => n_components=" + str(components))
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()

    # convert to dataframe, plot to see if we can identify clusters
    PCA_df = pd.DataFrame(p_components)
    plt.scatter(PCA_df[0], PCA_df[1], alpha=0.1, color="blue")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    # Apply KMeans
    km = KMeans(n_clusters=2)
    km_list = list(km.fit_predict(PCA_df))
    return km_list

def nrm_pca_kmeans(components):
    pca = PCA(n_components=components)
    p_components = pca.fit_transform(nrm)
    features = range(pca.n_components_)

    # Plot variance drop off of components, essentially why they were picked
    plt.bar(features, pca.explained_variance_ratio_, color='orange')
    plt.title("PCA => n_components="+str(components))
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()

    # convert to dataframe, plot to see if we can identify clusters
    PCA_df = pd.DataFrame(p_components)
    plt.scatter(PCA_df[0], PCA_df[1], alpha=0.1, color="blue")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

    # Apply KMeans
    km = KMeans(n_clusters=2, init="random", random_state=0)
    km_list = list(km.fit_predict(PCA_df))

    centroids = km.cluster_centers_
    plt.scatter(centroids[0][0], centroids[0][1], c=["blue"], edgecolors="black")
    plt.scatter(centroids[1][0], centroids[1][1], c=["red"], edgecolors="black")
    plt.scatter(PCA_df[0], PCA_df[1], alpha=0.05)
    plt.show()
    return km_list

# DBScan implementation
def dbscan(components, assign_to_forged):
    # Apply PCA, reduce dimensionality
    pca = PCA(n_components=components)
    p_components = pca.fit_transform(nrm)
    features = range(pca.n_components_)


    # convert to dataframe, plot to see if we can identify clusters
    PCA_df = pd.DataFrame(p_components)
    plt.scatter(PCA_df[0], PCA_df[1], alpha=0.1, color="blue")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()

    # Apply DBScan, need to test different eps values
    # Best for this dataset I have found so far... eps=0.2, min_samples=31
    dbs = DBSCAN(eps=0.2, min_samples=31)
    list_predictions = list(dbs.fit_predict(PCA_df))

    plt.scatter(PCA_df[0], PCA_df[1], c=list_predictions, cmap="Paired")
    plt.show()

    print("Number of Outliers's", list_predictions.count(-1))

    # Assign outliers to Forged or not
    legit_vals = [-1, 0, 1]
    if assign_to_forged:
        for i in range(len(list_predictions)):
            if list_predictions[i] == -1:
                list_predictions[i] = 1
            if list_predictions[i] not in legit_vals:
                list_predictions[i] = 0

    else:
        for i in range(len(list_predictions)):
            if list_predictions[i] == -1:
                list_predictions[i] = 0
            if list_predictions[i] not in legit_vals:
                list_predictions[i] = 0

    return flip_results(list_predictions)


# for submission to Kaggle
def get_percentages_to_csv(list_pred):
    percentage_0 = list_pred.count(0) / len(list_pred)
    percentage_1 = list_pred.count(1) / len(list_pred)

    print("Resulting list: ", list_pred)
    print("Percentage Genuine: ", percentage_0)
    print("Percentage Forged: ", percentage_1)

    returning_df = data.drop(["V1", "V2", "V3", "V4"], axis=1)
    returning_df["ID"] = [i for i in range(1, len(data)+1)]
    returning_df["Class"] = list_pred
    returning_df.to_csv("tst_kaggle_A5.csv", index=False)

def flip_results(the_list):
    for i in range(len(the_list)):
        if the_list[i] == 0:
            the_list[i] = 1
        else:
            the_list[i] = 0
    return the_list

"""
Uncomment to choose! (not very elegant i know...)
"""

# Default K_Means
#get_percentages_to_csv(k_means_nrm())
#get_percentages_to_csv(k_means_std())

# With PCA, various number of components
#std_pca_kmeans(4)
#std_pca_kmeans(2)
#std_pca_kmeans(3)

#get_percentages_to_csv(nrm_pca_kmeans(components=2))
#get_percentages_to_csv(nrm_pca_kmeans(components=2))
#get_percentages_to_csv(std_pca_kmeans(components=3))

#get_percentages_to_csv(dbscan(2, True))
#get_percentages_to_csv(dbscan_no_pca(True))
