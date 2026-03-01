import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from football_main_app import col_fotmob
from scipy.stats import poisson, skew
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

st.set_page_config(
    page_title='Poisson Predictions', 
    layout='wide', 

)

if st.session_state['logged_in']:
    col = col_fotmob

    #setting colors
    background = '#e1ece1'
    text = '#073d05'
    
    #filtering data
    YEAR = 2026
    SEASONS = [f"{YEAR}", f"{YEAR-1}/{YEAR}"]
    
    if st.session_state['user']['plan'] == 'free':
        leagues = ['LaLiga']
    else:
        leagues = col_fotmob.distinct('general.league')

    @st.cache_data(show_spinner=False, ttl='12h')
    def get_stats(year: int, leagues: list) -> dict:
        YEAR = year
        stats = list(col.aggregate([{"$match": {"general.league": {"$in": leagues}, "general.season": {"$in": SEASONS}}}, 
                                              {"$project": {"_id": 0, "general": 1, "teams": 1, "stats": 1, "score": 1, 'result': 1}}]))
        
        return stats
    
    def stats_to_df(stats: dict) -> pd.DataFrame:
    
        leagues = []
        seasons = []
        home = []
        home_images = []
        away = []
        away_images = []
        weighted_performances_home = []
        weighted_performances_away = []
        score_home = []
        score_away = []
        goals_sum = []
        results = []
    
        weights = np.array([1.12, 1.25, 1.32, 1.50])
    
    
        for stat in stats:
            #print(stat)
            leagues.append(f"{stat['general']['country']} - {stat['general']['league']}")
            seasons.append(stat['general']['season'])
            home.append(stat['teams']['home']['name'])
            home_images.append(stat['teams']['home']['image'])
            away.append(stat['teams']['away']['name'])
            away_images.append(stat['teams']['away']['image'])
    
            values_home = np.array([stat['stats']['ball_possession']['home'], stat['stats']['passes_opp_half_%']['home'], stat['stats']['touch_opp_box_100_passes']['home'], stat['stats']['xg_op_for_100_passes']['home']])
            
            weighted_performance_home = np.average(values_home, weights=weights, axis=0)
            weighted_performances_home.append(weighted_performance_home)
    
            values_away = np.array([stat['stats']['ball_possession']['away'], stat['stats']['passes_opp_half_%']['away'], stat['stats']['touch_opp_box_100_passes']['away'], stat['stats']['xg_op_for_100_passes']['away']])
            
            weighted_performance_away = np.average(values_away, weights=weights, axis=0)
            weighted_performances_away.append(weighted_performance_away)
            
            score_home.append(stat['score']['home'])
            score_away.append(stat['score']['away'])
            goals_sum.append(stat['score']['home'] + stat['score']['away'])
            results.append(stat['result'])
    
    
        df = pd.DataFrame({
            'league': leagues, 
            'season': seasons,
            'home': home, 
            'home_image': home_images,
            'away': away, 
            'away_image': away_images,
            'weighted_performance_home': weighted_performances_home, 
            'weighted_performance_away': weighted_performances_away, 
            'score_home': score_home, 
            'score_away': score_away, 
            'goals_sum': goals_sum,
            'result': results
    
        })
    
        return df
    
    def get_total_goals_avg(df: pd.DataFrame, column: str) -> float:
        total_goals_avg = df[column].mean()
    
        return total_goals_avg
    
    def get_rho(goals_avg: float) -> float:
        if goals_avg >= 3.0:
            rho = -0.02
        elif goals_avg <= 2.6:
            rho = -0.1
        else:
            rho = -0.05
    
        return rho
    
    def venue_goals_avg(df: pd.DataFrame, column_home: str, column_away: str) -> tuple[float]:
        home_goals_avg = df[column_home].mean()
        away_goals_avg = df[column_away].mean()
    
        return home_goals_avg, away_goals_avg
        
    @st.fragment
    def get_goals_metrics(df_league: pd.DataFrame, column_team_home: str, home_team: str, column_home_scores: str, columns_team_away: str, away_team: str, columns_away_scores: str, total_home_avg: float, total_away_avg: float) -> tuple[float]:
        home_scored_avg = df_league[df_league[column_team_home] == home_team][column_home_scores].mean()
        away_scored_avg = df_league[df_league[columns_team_away] == away_team][columns_away_scores].mean()
        
        home_conceced_avg = df_league[df_league[column_team_home] == home_team][columns_away_scores].mean()
        away_conceced_avg = df_league[df_league[columns_team_away] == away_team][column_home_scores].mean()    
        
        
        home_of_cap = home_scored_avg / total_home_avg
        away_of_cap = away_scored_avg / total_away_avg
        
        home_def_cap = home_conceced_avg / total_away_avg
        away_def_cap = away_conceced_avg / total_home_avg
    
        home_goals = home_of_cap * away_def_cap * total_home_avg
        away_goals = away_of_cap * home_def_cap * total_away_avg
    
        return home_goals, away_goals

    
    def get_dispersion_metrics(data: pd.DataFrame, score_home: str, score_away: str, goals_sum: str) -> tuple[np.float64]:
        score_home_mean = np.mean(data[score_home])
        score_home_var = np.var(data[score_home])
        score_away_mean = np.mean(data[score_away])
        score_away_var = np.var(data[score_away])
        total_goals_mean = np.mean(data[goals_sum])
        total_goals_var = np.var(data[goals_sum])
    
        d_home = np.round(score_home_var / score_home_mean, 2)
        d_away = np.round(score_away_var / score_away_mean, 2)
        d_skellmam = np.round(np.var(data[score_home] - data[score_away]) / total_goals_mean, 2)
        d_total_goals = np.round(total_goals_var / total_goals_mean, 2)
        skew_league = np.round(skew(data[score_home] - data[score_away]), 2)


        return d_home, d_away, d_skellmam, d_total_goals, skew_league
    
    def get_matrix_poisson(home_goals: float, away_goals: float, max_goals: int) -> np.outer:
        home_probs = [poisson.pmf(i, home_goals) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_goals) for i in range(max_goals)]
    
        goal_matrix = np.outer(home_probs, away_probs) 
    
        return goal_matrix
    
    ##### correction low scores
    def dc_tau(i, j, lam_home, lam_away, rho):
        if i == 0 and j == 0:
            return 1 - (lam_home * lam_away * rho)
        if i == 1 and j == 0:
            return 1 + (lam_away * rho)
        if i == 0 and j == 1:
            return 1 + (lam_home * rho)
        if i == 1 and j == 1:
            return 1 - rho
        return 1.0
    
    def apply_dixon_coles_to_matrix(P: np.outer, lam_home: float, lam_away: float, rho: float, normalize=True):
        P_dc = P.astype(float).copy()
        for i in (0, 1):
            for j in (0, 1):
                if i < P_dc.shape[0] and j < P_dc.shape[1]:
                    P_dc[i, j] *= dc_tau(i, j, lam_home, lam_away, rho)
    
        if normalize:
            s = P_dc.sum()
            if s <= 0:
                raise ValueError("The sum of the matrix after adjustment was ≤ 0. Check rho/lambdas.")
            P_dc /= s
    
        return P_dc
    
    def matrix_to_df(goal_matrix: np.outer, home_team: str, away_team: str) -> pd.DataFrame: 
        max_goals = goal_matrix.shape[0]
        
        goals_table = pd.DataFrame(
            goal_matrix, 
            columns=[f"{away_team} {i}" for i in range(max_goals)],
            index=[f"{home_team} {i}" for i in range(max_goals)]
        )
        
        goals_table = goals_table * 100
        
        return goals_table
    
    def get_match_probs(goal_matrix: np.outer) -> dict:  
        home_win_prob = np.tril(goal_matrix, k=-1).sum() * 100
        draw_prob = goal_matrix.diagonal().sum() * 100
        away_win_prob = np.triu(goal_matrix, k=1).sum() * 100
        
        home_double_chance_prob = np.tril(goal_matrix, k=0).sum() * 100
        away_double_chance_prob = np.triu(goal_matrix, k=0).sum() * 100
        any_win_double_chance_prob = home_win_prob + away_win_prob
        
        
        home_handicap_0 = home_win_prob / any_win_double_chance_prob * 100
        away_handicap_0 = away_win_prob / any_win_double_chance_prob * 100
    
        match_probs = {
            "Match Odds": 
            {'Home': {
                'Prob': home_win_prob, 
                'Odd': np.round(100 / home_win_prob, 2)
            },
            "Draw": {
                'Prob': draw_prob, 
                'Odd': np.round(100 / draw_prob, 2)
            }, 
            "Away": {
                'Prob': away_win_prob, 
                'Odd': np.round(100 / away_win_prob, 2)
            },
                }, 
            "Double Chance": {
                "Home": {
                    "Prob": home_double_chance_prob, 
                    "Odd": np.round(100  / home_double_chance_prob, 2)
                }, 
                 "No Draw": {
                    "Prob": any_win_double_chance_prob, 
                    "Odd": np.round(100  / any_win_double_chance_prob, 2)
                },
                "Away": {
                    "Prob": away_double_chance_prob, 
                    "Odd": np.round(100  / away_double_chance_prob, 2)
                }           
            }, 
            "Draw No Bet": {
                "Home": {
                    "Prob": home_handicap_0, 
                    "Odd": np.round(100  / home_handicap_0, 2)
                }, 
                "Away": {
                    "Prob": away_handicap_0, 
                    "Odd": np.round(100  / away_handicap_0, 2)
                }
            }
        }
        
        return match_probs
    
    def get_btts_probs(goal_matrix: np.outer) -> dict:     
        btts_yes = goal_matrix[1:, 1:].sum()
        btts_no = 1 - btts_yes
    
        btts = {
            "BTTS": {
                "Yes": {
                    "Prob": btts_yes * 100, 
                    "Odd": np.round(1 / btts_yes, 2)
                }, 
                "No": {
                    "Prob": btts_no * 100, 
                    "Odd": np.round(1 / btts_no, 2)
                }
            }
        }
    
        return btts
    
    def get_unders_overs(home_goals: float, away_goals: float) -> dict:
        goals_probs = {}
        goals = home_goals + away_goals
    
        for i in range(0, 6):
            under = poisson.cdf(i, goals)
            over = 1 - under
            goals_probs[f"{i}.5"] = {
                "Over": {
                    "Prob": over * 100, 
                    "Odd": np.round(1 / over, 2)
                }, 
                "Under": {
                    "Prob": under * 100, 
                    "Odd": np.round(1 / under, 2)
                }
            }
    
        return goals_probs
    
    def probs_to_df(probs: dict) -> pd.DataFrame: 
        reform = {(outerKey, innerKey): values for outerKey, innerDict in probs.items() for innerKey, values in innerDict.items()}
        df = pd.DataFrame.from_dict(reform, orient='index').transpose()
        df.columns = pd.MultiIndex.from_tuples(df.columns)
    
        return df.round(2)
    
    def input_form(home_teams: list, away_teams: list):
        with st.form(key='poisson'):
            if 'home' not in st.session_state:
                st.session_state['home'] = home_teams
            
            if 'away' not in st.session_state:
                st.session_state['away'] = away_teams
        
            col1, col2 = st.columns(2)
        
            with col1:
                home = st.selectbox(label="Select a Home Team", options=home_teams, index=0)
            
            with col2:
                away = st.selectbox(label="Select an Away Team", options=away_teams, index=1)
        
            submitted = st.form_submit_button("Submit")
    
        return submitted, home, away

    def ideal_len(data: pd.DataFrame, home_col: str = 'home', away_col: str = 'away') -> bool:
        home_teams = data[home_col].unique()
        away_teams = data[away_col].unique()

        all_teams = list(set(home_teams) | set(away_teams))

        min_matches = len(all_teams) * 10

        return min_matches
    
    def style_df(df: pd.DataFrame) -> pd.DataFrame:
        styled_df = df.style.highlight_between(
            color='#5A915A',
            left=1.7,
            right=2,
            axis=1,
            subset=pd.IndexSlice[['Odd'], :]
        ).format(precision=2)
    
        return styled_df
    
    def create_colormap(colors: list) -> LinearSegmentedColormap: 
        custom_camp = LinearSegmentedColormap.from_list('my_colormap', colors=colors, N=256)
    
        return custom_camp
    
    def plot_venue_performances(df: pd.DataFrame, home_team: str, away_team: str) -> None:
        df_home = df[df['home'] == home_team]
        df_away = df[df['away'] == away_team]
    
        fig,ax = plt.subplots(figsize=(8, 5))
        ax.set_title('Weighted Performance Home/Away', fontweight='bold')
        ax.set_xlabel(' ')
        ax.set_ylabel('Weighted Performance')
        sns.barplot(data=df_home, x='home', y='weighted_performance_home', ax=ax, color='#213991')
        sns.barplot(data=df_away, x='away', y='weighted_performance_away', ax=ax, color='#915221')
        sns.despine()
        ax.set_facecolor("#e3f5ea")
        fig.set_facecolor("#e3f5ea")
    
        return st.pyplot(fig)
    
    
    stats = get_stats(year=YEAR, leagues=leagues)
    df = stats_to_df(stats=stats)
    
    st.title('Poisson Probabilities')
    league = st.selectbox(label='Select a League:', options=sorted(df['league'].unique()), index=0)
    df_league = df[df['league'] == league]
    
    st.dataframe(df_league.drop(columns=['home_image', 'away_image']), hide_index=True)
    if ideal_len(df_league) > len(df_league):
        st.warning(f"{league} still does not have enough matches for reliable predictability using the Poisson distribution. It is recommended to wait until there are more rounds played.", icon="🚨")
    home_teams = df_league['home'].unique()
    away_teams = df_league['away'].unique()

    d_home, d_away, d_skellmam, d_total_goals, skew_league = get_dispersion_metrics(data=df_league, score_home='score_home', score_away='score_away', goals_sum='goals_sum')
    st.subheader(f"{league} Dispersion Metrics")
    col_home, col_away, col_skellmam, col_total_goals, col_skew = st.columns(5)
    
    with col_home:
        st.metric(label='Home Goals Dispersion', value=d_home, border=True)
        if d_home > 1.3:
            st.warning("High home-goal volatility detected. Large home scorelines may occur more often than the Poisson model expects.")
        elif d_home > 1.1:
            st.info("Moderate home-goal variability. Slight tendency for bigger home wins than predicted.")
        elif d_home < 0.9:
            st.info("Low home-goal variability. Home scoring is more controlled than the Poisson assumption.")
        else:
            st.success("Home goals behave close to a Poisson process.")
    
    with col_away:
        st.metric(label='Away Goals Dispersion', value=d_away, border=True)
        if d_away > 1.3:
            st.warning("High away-goal volatility detected. Rare high away scores may be more frequent than expected.")
        elif d_away > 1.1:
            st.info("Moderate away-goal variability. Slightly more dispersion than Poisson assumes.")
        elif d_away < 0.9:
            st.info("Low away-goal variability. Away scoring is more concentrated than expected.")
        else:
            st.success("Away goals behave close to a Poisson process.")
    
    with col_skellmam:
        st.metric(label='Margin Dispersion (Skellmam)', value=d_skellmam, border=True)
        if d_skellmam > 1.3:
            st.warning("High winning-margin volatility detected. Large victories and heavy defeats are more frequent than expected.")
        elif d_skellmam > 1.1:
            st.info("Moderate winning-margin variability. Slightly wider margins than the model assumes.")
        elif d_skellmam < 0.9:
            st.info("Low winning-margin variability. Matches tend to be decided by narrow margins.")
        else:
            st.success("Winning margins behave close to the Skellam (Poisson-based) expectation.")
    
    with col_total_goals:
        st.metric(label='Total Goal Dispersion', value=d_total_goals, border=True)
        if d_total_goals > 1.3:
            st.warning("High total-goal volatility detected. Extreme scorelines are more frequent than the Poisson model expects.")
        elif d_total_goals > 1.1:
            st.info("Moderate total-goal variability. Slightly more extreme totals may occur.")
        elif d_total_goals < 0.9:
                st.info("Low total-goal variability. Matches tend to be more controlled than the Poisson assumption.")
        else:
            st.success("Total goals behave close to a Poisson process.")
    with col_skew:
        st.metric(label='Winning Margin Asymmetry (Skew)', value=skew_league, border=True)
        if skew_league > 0.5:
            st.warning("Strong positive skew detected. Extreme results tend to favor large home victories.")
        elif skew_league > 0.2:
            st.info("Mild positive skew. Big home wins occur slightly more often than big away wins.")
        elif skew_league < -0.5:
            st.warning("Strong negative skew detected. Extreme results tend to favor large away victories.")
        elif skew_league < -0.2:
            st.info("Mild negative skew. Big away wins occur slightly more often.")
        else:
            st.success("Scoreline extremes are relatively symmetric between home and away.")
    
    
    
    st.markdown("""
    Want to understand what these numbers mean in practice? Read the full breakdown here: 👉 [How to Read Dispersion and Skew in Football Models](https://open.substack.com/pub/saulofaria/p/how-to-read-dispersion-and-skew-in?r=30n7hp&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
    """)
    
        
    submitted, home, away = input_form(home_teams=home_teams, away_teams=away_teams)
        
    if submitted: 
        total_home_avg, total_away_avg = venue_goals_avg(df=df_league, column_home='score_home', column_away='score_away')
        home_goals, away_goals = get_goals_metrics(df_league=df_league, column_team_home='home', home_team=home, column_home_scores='score_home', columns_team_away='away', away_team=away, columns_away_scores='score_away', 
                                               total_home_avg=total_home_avg, total_away_avg=total_away_avg)
        
        total_goals_avg = get_total_goals_avg(df=df_league, column='goals_sum')
        rho = get_rho(goals_avg=total_goals_avg)
        goal_matrix = get_matrix_poisson(home_goals=home_goals, away_goals=away_goals, max_goals=7)
        goal_matrix = apply_dixon_coles_to_matrix(P=goal_matrix, lam_home=home_goals, lam_away=away_goals, rho=rho)
    
        match_probs = get_match_probs(goal_matrix=goal_matrix)
        btts = get_btts_probs(goal_matrix=goal_matrix)
        goal_probs = get_unders_overs(home_goals=home_goals, away_goals=away_goals)
    
        goals_table = matrix_to_df(goal_matrix=goal_matrix, home_team=home, away_team=away)
        match_odds_df = probs_to_df(match_probs)
        btts_df = probs_to_df(btts)
        goal_df = probs_to_df(goal_probs)
    
        custom_cmap = create_colormap(colors=["#2F3B2F", "w", "#15E615"])
    
        st.header('Poisson Heatmap and Weighted Perfomances Home/Away')
        st.text('Weighted Performances are based on metrics such as Open-Play xG Per 100 Passes and so on.')
    
        col1, col2 = st.columns([9, 4])
    
        with col1:
            st.dataframe(goals_table.style.background_gradient(cmap=custom_cmap, axis=None).format(precision=2))
        with col2:
            plot_venue_performances(df=df_league, home_team=home, away_team=away)
    
        st.divider()
    
        st.header('Probabilities and Odds')
        st.text('Odds between 1.70 and 2.00 stand out because, historically, they have delivered the highest ROI in the long run. But the decision is up to you.')
        st.warning('These are the fair odds according to the Poisson distribution. They are not necessarily the odds offered by the bookmakers.', icon="⚠️")
        col3, col4 = st.columns([10, 3])
    
        with col3:
            st.subheader('Result Probabilities')
            st.dataframe(style_df(match_odds_df))
        with col4:
            st.subheader('BTTS Probabilities')
            st.dataframe(style_df(btts_df))
    
        st.subheader('Overs and Unders Probabilities')
        st.dataframe(style_df(goal_df))
        
