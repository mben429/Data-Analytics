from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style

"""
Goal here, is to get NBA stats!!
"""

scoring_leaders_url = "https://www.basketball-reference.com/leaders/pts_per_g_yearly.html"


def scrape_url(curr_url):
    uClient = uReq(curr_url)
    page_html = uClient.read()
    uClient.close()

    page_soup = soup(page_html, "html5lib")

    players = page_soup.findAll('tr')
    players.pop(0)
    player_names, player_ppgs, player_teams, years = [], [], [], []
    print(players[0])
    print("\n")

    # Loop through and extract all the important information
    for player in players:
        player_names.append(str(player.contents[5].a.next_element))  # contents gets children

        player_team = player.contents[8]
        player_teams.append(str(player_team.next_element.next_element))

        years.append(str(player.td)[4:11])

        player_ppg = float(str(player.contents[6])[4:9])
        player_ppgs.append(player_ppg)

    return player_names, years, player_teams, player_ppgs


def get_count_teams(teams):
    unique_teams = set(teams)
    count_of_teams = {}

    for team in unique_teams:
        if team not in count_of_teams:
            count_of_teams[team] = teams.count(team)

    # Order by value
    tup_in_order = list(sorted(count_of_teams.items(), key=lambda x: x[1], reverse=True))

    # Get top 10
    tup_in_order = tup_in_order[:10]

    return tup_in_order


def plot_leading_scoring_teams(teams_count):
    y_list = [x[1] for x in teams_count]
    x_list = [x[0] for x in teams_count]

    barlist = plt.bar(x_list, y_list, align="center", alpha=0.85, color="red")

    plt.xlabel("Team")
    plt.ylabel("Number of years having the leading scorer in the NBA")
    plt.show()


# Lists of stats
names, years, teams, ppg = scrape_url(scoring_leaders_url)

# Get count of teams with Leading scorers
teams_count_list = get_count_teams(teams)
print(teams_count_list)

# Plot graph
plot_leading_scoring_teams(teams_count_list)
