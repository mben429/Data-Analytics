import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go

years = [i for i in range(1987, 2023, 4)]

# Getting data
first_points = [126, 68, 112, 102, 113, 105, 62, 97, 69]

plt.bar(x=years,
        height=first_points,
        width=2.5,
        align="center",
        color=["black", "green", "blue", "lightblue", "red", "darkgreen", "darkgreen", "lightblue", "darkgreen"],
        edgecolor=["black", "orange", "red", "yellow", "blue", "gold", "gold", "yellow", "gold"]
)
plt.xticks(years)
plt.title("Top point scorer in each tournament")
plt.xlabel("RWC Year")
plt.ylabel("Points")
plt.show()