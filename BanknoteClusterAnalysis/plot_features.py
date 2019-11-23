import pandas as pd
import matplotlib.pyplot as plt


raw_data = pd.read_csv("BanknoteData.csv")

print(raw_data.head())
titles = ["Entropy", "Skewness", "Variance", "Kurtosis"]
features = ["V1", "V2", "V3", "V4"]


def plot_distributions(feature, title):
    pd.DataFrame.hist(raw_data, feature, grid=False)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel(title + " Measure")
    plt.show()


for i in range(len(features)):
    plot_distributions(features[i], titles[i])
