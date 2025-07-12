# League of Legends Challenger Analysis: Paths to the Top
**Summer 2025 Data Science Project**  
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

---

## 2. Introduction

This analysis investigates the paths to reaching **Challenger rank** in *League of Legends*, the highest competitive tier with only 300 slots per region.

### Research Questions:
- What are the main characteristics that distinguish Challenger players?
- Is there a correlation between win rate and League Points?
- Are there different “archetypes” or paths to reaching Challenger?
- Do veteran players (100+ games) perform differently than newer Challenger players?

### Why This Matters:
Understanding the patterns of elite players provides insight into:
- Competitive gaming success
- Skill progression strategies
- Game balance and design

This study explores whether success is driven more by *exceptional win rates* or *grinding a large number of games*.

---

## 3. Data Curation

### Data Source:
- Riot Games API – [https://developer.riotgames.com](https://developer.riotgames.com)

### Collection Process:
- Queried the **North American Challenger League endpoint** to retrieve data on all 300 Challenger players.
- For each player:
  - Summoner info
  - Rank details
  - Wins and losses
  - Status flags (veteran, hot streak, fresh blood, inactive)

### API Compliance:
- Rate limiting:
  - 100 requests per 2 minutes
  - 20 requests per second
- Total API calls: **301**

### Dataset Summary:
- **Rows**: 300 players
- **Columns**: 15 features  
- **Key Variables**:
  - `league_points`: LP (range 795–2095)
  - `win_rate`: Wins / (Wins + Losses)
  - `total_games`: Wins + Losses
  - `veteran`: 100+ games in Challenger
  - `hot_streak`, `fresh_blood`: performance flags

### Preprocessing Example:
```python
df_players['total_games'] = df_players['wins'] + df_players['losses']
df_players['win_rate'] = df_players['wins'] / df_players['total_games']
df_players['games_per_lp'] = df_players['total_games'] / df_players['league_points']
```

4. Exploratory Data Analysis
Key Statistics:
Average win rate: 55.7% ± 3.5%

League Points range: 795 – 2095 (mean: 1095)

Games played: 131 – 1256 (mean: 452 ± 207)

Distribution Analysis:
Both the win rate and league points (LP) distributions are right-skewed. This suggests that while most Challenger players hover near the mean, a smaller subset of extremely high-performing players boosts the upper end of the distributions.


Correlation Analysis:
The Pearson correlation matrix shows:

Total games ↔ Wins: r = 0.997 (expected)

Total games ↔ Losses: r = 0.997 (expected)

Total games ↔ Win rate: r = -0.692 (p < 0.0001)


These results reveal an inverse relationship between win rate and total games played, hinting at two distinct “paths” to Challenger:

High win rate and fewer games

Moderate win rate and high volume

5. Primary Analysis
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

6. Visualization
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

7. Insights and Conclusions
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
