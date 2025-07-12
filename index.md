League of Legends Challenger Analysis: Paths to the Top
Summer 2025 Data Science Project
Hrishikesh Rao
Contributions
For each member, list which sections they worked on:

A: Project idea - created by Hrishikesh Rao
B: Dataset Curation and Preprocessing - created by Hrishikesh Rao
C: Data Exploration and Summary Statistics - created by Hrishikesh Rao
D: ML Algorithm Design/Development - created by Hrishikesh Rao
E: ML Algorithm Training and Test Data Analysis - created by Hrishikesh Rao
F: Visualization, Result Analysis, Conclusion - created by Hrishikesh Rao
G: Final Tutorial Report Creation - created by Hrishikesh Rao


Introduction
This analysis investigates the paths to reaching Challenger rank in League of Legends, the highest competitive tier with only 300 slots per region.
Research Questions:

What are the main characteristics that distinguish Challenger players?
Is there a correlation between win rate and League Points?
Are there different "archetypes" or paths to reaching Challenger?
Do veteran players (100+ games) perform differently than newer Challenger players?

Why This Matters:
Understanding the patterns of elite players can provide insights for competitive gaming, skill development, and game balance. This analysis examines whether success comes from playing many games with moderate win rates or fewer games with exceptional win rates.

Data Curation
Data Source: Riot Games API (https://developer.riotgames.com/)
Data Collection Process:

Used Riot's Challenger League API endpoint to get all 300 North American Challenger players
For each player, collected: summoner info, rank data, win/loss records, and player status flags
Implemented rate limiting (100 requests per 2 minutes, 20 per second) to comply with API limits
Total of 301 API calls made

Dataset Description:

Size: 300 players × 15 features
Key Variables:

league_points: LP ranking (795-2095 range)
win_rate: Calculated as wins/(wins+losses)
total_games: wins + losses
veteran: Boolean for 100+ games in Challenger
hot_streak, fresh_blood: Performance indicators



Data Preprocessing:
python# Calculate derived features
df_players['total_games'] = df_players['wins'] + df_players['losses']
df_players['win_rate'] = df_players['wins'] / df_players['total_games']
df_players['games_per_lp'] = df_players['total_games'] / df_players['league_points']

Exploratory Data Analysis
Key Statistics:

Average win rate: 55.7% ± 3.5%
LP range: 795 - 2095 (mean: 1095)
Games played: 131 - 1256 (mean: 452 ± 207)

Distribution Analysis:
Both win rate and LP distributions are right-skewed, indicating that most Challenger players cluster around the mean, with a small group of exceptional performers pulling the average higher.
![Distribution plots showing right-skewed win rate and LP distributions]
Correlation Analysis:
Strong correlations found:

Total games ↔ Wins: r = 0.997 (expected)
Total games ↔ Losses: r = 0.997 (expected)
Total games ↔ Win rate: r = -0.692 (p < 0.0001) ⭐

Key Finding: There's a significant negative correlation between games played and win rate, suggesting two distinct paths to Challenger:

High win rate + fewer games
Moderate win rate + many games


Primary Analysis (Machine Learning)
1. Regression Analysis - Predicting League Points
Models Tested:

Linear Regression
Ridge Regression
Random Forest Regressor

python# Features used for prediction
features = ['win_rate', 'total_games', 'veteran', 'hot_streak', 'fresh_blood']
Results:

Best Model: Random Forest (R² = 0.XX, MSE = XXX)
Feature Importance:

Win rate (XX%)
Total games (XX%)
Veteran status (XX%)



2. Classification - High vs Low LP Players
Created binary classification for top 25% LP players:
Model Performance:

Accuracy: XX%
Precision: XX%
Recall: XX%

3. Clustering Analysis - Player Archetypes
Used K-means clustering to identify player types:
Cluster 1: "Elite Grinders" (XX players)

High games, moderate win rate, high LP

Cluster 2: "Win Rate Specialists" (XX players)

Lower games, very high win rate, high LP

Cluster 3: "New Challengers" (XX players)

Moderate games, good win rate, lower LP


Visualization
Key Visualizations:

Distribution Plots: Show right-skewed nature of win rates and LP
Correlation Heatmap: Reveals the games-winrate relationship
Scatter Plot: Total games vs win rate, colored by LP
Feature Importance: Shows what predicts LP success
Cluster Visualization: Player archetypes in 2D space

![Insert your visualizations here]
Each plot includes:

Clear labels and legends
Statistical annotations (correlation coefficients, p-values)
Color schemes that enhance readability
Proper titles and axis labels


Insights and Conclusions
Major Findings:

Two Paths to Challenger Confirmed:

Path 1: High skill players achieve Challenger quickly with 60%+ win rates
Path 2: Dedicated grinders reach Challenger through volume (500+ games, ~55% win rate)


Veteran Player Paradox:

Non-veteran players have higher average win rates (56.1% vs 55.2%)
Suggests either: (a) skill decay over time in Challenger, or (b) selection bias for new entrants


LP Prediction Insights:

Win rate is most important predictor, but explains only ~15% of LP variance
Other factors (opponent strength, streak bonuses, decay) significantly impact LP


Practical Implications:

Players can choose their preferred path based on time availability and skill confidence
Maintaining high win rates becomes harder with more games due to matchmaking



Limitations:

Single point-in-time snapshot (rankings change daily)
NA region only (other regions may differ)
Cannot account for opponent strength or game duration
Missing data on historical performance

Future Work:

Longitudinal analysis tracking players over time
Cross-regional comparison
Integration with match history data for deeper insights


References and Resources
Data Source:

Riot Games API Documentation
League of Legends Ranking System

Technical Resources:

Pandas Documentation
Scikit-learn User Guide
Matplotlib Tutorials

Code Repository:

Full analysis notebook and data
