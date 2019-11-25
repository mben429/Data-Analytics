from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as soup
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style


rank_years_range = [i for i in range(2004, 2020)]

usa = [15, 16, 14, 14, 19, 19, 16, 16, 17, 16, 18, 16, 16, 17, 17, 13]

fig, ax = plt.subplots()

ax.plot(rank_years_range, usa, label="USA", color="red")
ax.legend()
plt.title("IRB World ranking progression - USA")
plt.xlabel("Year")
plt.ylabel("IRB World Rank")
plt.gca().invert_yaxis()
plt.show()
