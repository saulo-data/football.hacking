import numpy as np 
import pandas as pd
import streamlit as st
from db_conn import db
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title='Tables Predictions', 
    layout='wide', 

)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state['logged_in']:
    col_fotmob = db.fotmob_stats
    col_matches = db.fotmob_next_matches 
    col_table = db.fotmob_tables 
    
    #filtering data
    YEAR = 2026
    SEASONS = [f"{YEAR}", f"{YEAR-1}/{YEAR}"]
    
    if st.session_state['user']['plan'] == 'free':
       leagues_filter = ('LaLiga',)   # tuple
    else:
       leagues_filter = tuple(col_fotmob.distinct('general.league'))

    @st.cache_data(show_spinner=False, ttl='12h')
    def get_stats(year: int, leagues_filter: tuple[str, ...]) -> dict:
        stats = list(db.fotmob_stats.aggregate([
            {"$match": {
                "general.season": {"$in": [f"{year}", f"{year-1}/{year}"]},
                "general.league": {"$in": list(leagues_filter)}
            }},
            {"$project": {"_id": 0, "general": 1, "teams": 1, "score": 1}}
        ]))
        
        return stats

    def stats_to_df(stats: dict) -> pd.DataFrame:
        leagues = []
        home = []
        away = []
        score_home = []
        score_away = []


        for stat in stats:
            leagues.append(f"{stat['general']['country']} - {stat['general']['league']}")
            home.append(stat['teams']['home']['name'])
            away.append(stat['teams']['away']['name'])        
            score_home.append(stat['score']['home'])
            score_away.append(stat['score']['away'])

        df = pd.DataFrame({
            'league': leagues, 
            'home': home, 
            'away': away, 
            'score_home': score_home, 
            'score_away': score_away
        })

        return df

    def split_strings(full_name: str) -> tuple[str]:
        country, league = full_name.split(' - ')

        return country, league


    @st.cache_data(show_spinner=False, ttl="12h")
    def get_stats_tables(country: str, league: str):
        doc = next(
            col_table.aggregate([
                {"$match": {"league_country": country, "league_name": league}},
                {"$project": {"_id": 0, "league_id": 1, "teams": 1}},
                {"$limit": 1},
            ]),
            None
        )
        if doc is None:
            return None, None
        return doc["league_id"], doc["teams"]

    def get_next_matches(league_id: int, home_col: str = 'home_team', away_col: str = 'away_team') -> pd.DataFrame:
        next_matches_db = list(col_matches.find({"league_id": league_id})) 
        next_matches = pd.DataFrame(next_matches_db)[[home_col, away_col]]

        return next_matches


    def monte_carlo_table(
        next_matches: pd.DataFrame,      
        current_points: pd.DataFrame,    
        df_league: pd.DataFrame,  
        n_sims: int = 50000,
        seed: int = 42,
    ):

        rng = np.random.default_rng(seed)
        teams = list(current_points["team_name"].values)
        n_teams = len(teams)
        team_to_idx = {t: i for i, t in enumerate(teams)}

        cur_points = (
            current_points.set_index("team_name")["points"]
            .reindex(teams)
            .fillna(0)
            .to_numpy(dtype=np.int32)
        )
        total_home_avg = df_league["score_home"].mean()
        total_away_avg = df_league["score_away"].mean()

        def get_goals_metrics(home_team: str, away_team: str):

            home_scored_avg = df_league[df_league["home"] == home_team]["score_home"].mean()
            away_scored_avg = df_league[df_league["away"] == away_team]["score_away"].mean()

            home_conceded_avg = df_league[df_league["home"] == home_team]["score_away"].mean()
            away_conceded_avg = df_league[df_league["away"] == away_team]["score_home"].mean()

            if pd.isna(home_scored_avg): home_scored_avg = total_home_avg
            if pd.isna(away_scored_avg): away_scored_avg = total_away_avg
            if pd.isna(home_conceded_avg): home_conceded_avg = total_away_avg
            if pd.isna(away_conceded_avg): away_conceded_avg = total_home_avg

            home_of_cap = home_scored_avg / total_home_avg
            away_of_cap = away_scored_avg / total_away_avg

            home_def_cap = home_conceded_avg / total_away_avg
            away_def_cap = away_conceded_avg / total_home_avg

            home_goals = home_of_cap * away_def_cap * total_home_avg
            away_goals = away_of_cap * home_def_cap * total_away_avg

            return max(home_goals, 0), max(away_goals, 0)

        n_matches = len(next_matches)

        home_idx = next_matches["home_team"].map(team_to_idx).to_numpy(dtype=np.int32)
        away_idx = next_matches["away_team"].map(team_to_idx).to_numpy(dtype=np.int32)

        lam_home = np.zeros(n_matches)
        lam_away = np.zeros(n_matches)

        for i, row in enumerate(next_matches.itertuples(index=False)):
            lh, la = get_goals_metrics(row.home_team, row.away_team)
            lam_home[i] = lh
            lam_away[i] = la

        home_goals = rng.poisson(lam_home, size=(n_sims, n_matches))
        away_goals = rng.poisson(lam_away, size=(n_sims, n_matches))

        home_points = np.where(home_goals > away_goals, 3,
                        np.where(home_goals == away_goals, 1, 0)).astype(np.int16)

        away_points = np.where(away_goals > home_goals, 3,
                        np.where(home_goals == away_goals, 1, 0)).astype(np.int16)

        points = np.zeros((n_sims, n_teams), dtype=np.int32)

        for m in range(n_matches):
            points[:, home_idx[m]] += home_points[:, m]
            points[:, away_idx[m]] += away_points[:, m]

        points += cur_points

        return points, teams


    def position_matrix(points_sims: np.ndarray, teams: list[str]) -> pd.DataFrame:

        n_sims, n_teams = points_sims.shape

        ranks = np.argsort(np.argsort(-points_sims, axis=1), axis=1) + 1

        pos_matrix = np.zeros((n_teams, n_teams))

        for t in range(n_teams):
            counts = np.bincount(ranks[:, t], minlength=n_teams + 1)
            pos_matrix[t, :] = counts[1:] / n_sims

        df = pd.DataFrame(
            pos_matrix,
            index=teams,
            columns=[f"{i}º" for i in range(1, n_teams + 1)]
        ) * 100

        return df.round(2)

    def percent_to_odds_df(df_percent: pd.DataFrame, cap: float = 10000.0):
        p = df_percent.astype(float).to_numpy() / 100.0
        odds = np.empty_like(p, dtype=float)
        odds[:] = np.nan
        mask = p > 0
        odds[mask] = 1.0 / p[mask]
        odds = np.clip(odds, None, cap)

        return pd.DataFrame(odds, index=df_percent.index, columns=df_percent.columns).round(2)

    def create_colormap(colors: list) -> LinearSegmentedColormap: 
        custom_camp = LinearSegmentedColormap.from_list('my_colormap', colors=colors, N=1000)

        return custom_camp

    def _prep_df_probs(df_probs: pd.DataFrame) -> pd.DataFrame:
        df = df_probs.copy()
    
        # columns -> int positions 1..N
        df.columns = [int(c) for c in df.columns]
        df = df.sort_index(axis=1)
    
        # fill/validate
        if df.isna().any().any():
            raise ValueError("df_probs contains NaN.")
        if (df.values < 0).any():
            raise ValueError("df_probs contains negative probabilities.")
    
        # normalize rows if needed
        rs = df.sum(axis=1).values
        if not np.allclose(rs, 1.0, atol=1e-6):
            df = df.div(df.sum(axis=1), axis=0)
    
        return df

    def table_metrics_from_df_probs(
        df_probs: pd.DataFrame,
        current_positions: pd.Series | dict | None = None,
    ):
        """
        Returns 4 metrics:
          1) Table Stability Index (TSI): mean prob of finishing in current position
          2) Avg Position Shift (EPS): mean expected |final_pos - current_pos|
          3) Rank Volatility (SD): mean std-dev of final position distribution
          4) Entropy (normalized 0..1): mean normalized Shannon entropy of position distribution
        """
        df = _prep_df_probs(df_probs)
        teams = df.index
        positions = np.array(df.columns, dtype=int)
        N = len(positions)
    
        # current positions:
        # - if not provided: assume df index order equals current table order (1..N)
        if current_positions is None:
            cur_pos = pd.Series(np.arange(1, N + 1), index=teams, dtype=int)
        else:
            cur_pos = pd.Series(current_positions, index=teams, dtype=int).reindex(teams)
            if cur_pos.isna().any():
                missing = cur_pos[cur_pos.isna()].index.tolist()
                raise ValueError(f"Missing current positions for teams: {missing}")
    
        P = df.to_numpy(dtype=float)
        pos_to_col = {p: j for j, p in enumerate(positions)}
    
        # 1) Table Stability Index (mean diagonal by current position)
        diag = np.array([P[i, pos_to_col[int(cur_pos.iloc[i])]] for i in range(len(teams))], dtype=float)
        TSI = float(diag.mean()) * 100
    
        # 2) Avg Position Shift (EPS): E[|final - current|]
        abs_dist = np.abs(positions.reshape(1, -1) - cur_pos.values.reshape(-1, 1))
        EPS_i = (P * abs_dist).sum(axis=1)
        EPS = float(EPS_i.mean())
    
        # 3) Rank Volatility (SD): std-dev of final position distribution
        mu_i = (P * positions.reshape(1, -1)).sum(axis=1)
        var_i = (P * (positions.reshape(1, -1) - mu_i.reshape(-1, 1)) ** 2).sum(axis=1)
        SD_i = np.sqrt(var_i)
        SD = float(SD_i.mean())
    
        # 4) Entropy (normalized 0..1)
        eps = 1e-15
        H_i = -(P * np.log(P + eps)).sum(axis=1)
        Hnorm_i = H_i / np.log(N)
        Hnorm = float(Hnorm_i.mean())
    
        return {
            "table_stability_index": TSI,
            "avg_position_shift": EPS,
            "rank_volatility_sd": SD,
            "entropy_norm": Hnorm,
            # optional per-team outputs (useful for UI)
            "per_team": pd.DataFrame(
                {
                    "current_pos": cur_pos.values,
                    "p_same_pos": diag,
                    "eps_shift": EPS_i,
                    "rank_sd": SD_i,
                    "entropy_norm": Hnorm_i,
                    "expected_pos": mu_i,
                },
                index=teams,
            ).sort_values("current_pos")
        }


    @st.cache_data(show_spinner=False, ttl="12h")
    def table_leagues_set():
        rows = list(col_table.find({}, {"_id": 0, "league_country": 1, "league_name": 1}))
        return {
            f"{r['league_country']} - {r['league_name']}"
            for r in rows
            if r.get("league_country") and r.get("league_name")
        }


    FH_DARK = LinearSegmentedColormap.from_list(
        "fh_dark",
        ["#f8fdf9", "#6ea889", "#041f10"] 
    )

    def probs_pct_to_odds_df(
        df_prob_pct: pd.DataFrame,
        start_col: int = 2,        
        min_prob_pct: float = 0.0, 
        max_odd: float = 200.0     
    ) -> tuple[pd.DataFrame, list]:
        """
        Converte colunas de probabilidade em % (0-100) para odds decimais.
        Retorna (df_odds, cols_odds).
        """
        df = df_prob_pct.copy()
        cols = list(df.columns[start_col:])

        
        p = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")

       
        p = p.mask(p <= min_prob_pct, np.nan)

        
        odds = 100.0 / p

        
        odds = odds.replace([np.inf, -np.inf], np.nan)
        odds = odds.mask(odds > max_odd, np.nan)

        df.loc[:, cols] = odds
        return df, cols

    
    def style_odds_df(
        df_odds: pd.DataFrame,
        odd_cols: list,
        cmap=FH_DARK,
        vmin_log: float | None = None,
        vmax_log: float | None = None,
        decimals: int = 2
    ):
        """
        Colorir odds de forma robusta:
        - gmap = log(odds) (melhor distribuição de cores)
        - cmap invertido: odds menores (mais prováveis) mais escuras
        - NaN fica vazio
        """
       
        g = df_odds.loc[:, odd_cols].apply(pd.to_numeric, errors="coerce")
        g = g.mask(g <= 0, np.nan)
        gmap = np.log(g)

        sty = (
            df_odds.style
            .background_gradient(
                cmap=cmap.reversed(),              
                axis=None,
                subset=pd.IndexSlice[:, odd_cols],
                gmap=gmap,
                vmin=vmin_log,
                vmax=vmax_log
            )
            .format(f"{{:.{decimals}f}}", subset=odd_cols, na_rep="")
        )
        return sty

    stats = get_stats(year= YEAR, leagues_filter=leagues_filter)
    df = stats_to_df(stats=stats)   


    valid = table_leagues_set()
    options = sorted(
        set(df["league"].dropna().astype(str).str.strip().unique().tolist()) & valid
    )

    if not options:
        st.error("Sem ligas disponíveis.")
        st.stop()

    key = "tables_preds__league"    
    prev = st.session_state.get(key)
    if prev not in options:
        st.session_state[key] = options[0]

    st.title("Monte Carlo Season Forecast")
    st.info("These probabilities come from Monte Carlo simulations of the remaining fixtures.")
    st.markdown("""
    This table shows the **probability of each team finishing in every league position** based on Monte Carlo simulations of the remaining matches.

    Each cell represents how often a team finished in that position across thousands of simulated seasons.

    **Darker cells indicate higher probabilities.**
    """)

    league_selected = st.selectbox("Select a League:", options=options, key=key)


    df_table = df[df['league'] == league_selected]
    country, league_name = split_strings(league_selected)
    league_id, teams = get_stats_tables(country, league_name)

    if league_id is None:
        st.warning(f"Sem tabela no banco para {league_selected}. Selecione outra liga.")
        st.stop()

    current_points = pd.json_normalize(teams)
    next_matches = get_next_matches(league_id=league_id)


    points_sims, teams = monte_carlo_table(next_matches, current_points, df_table, n_sims=50_000, seed=7)
    pos_df = position_matrix(points_sims, teams)
    pos_df2 = pos_df.copy()
    pos_df2.insert(0, "current_points", current_points.set_index("team_name").loc[teams, "points"].astype(str))
    pos_df2.insert(1, "pos", current_points.set_index("team_name").loc[teams, "pos"].astype(str))
    pos_df2['pos'] = pos_df2['pos'].apply(lambda x: f"{x}º")
    pos_df2 = pos_df2.rename(columns={'current_points': 'Current Pts', 'pos': 'Pos'})

    
    MAX_ODD = 200.0   
    MIN_PROB = 0.0    

    odds_df, odd_cols = probs_pct_to_odds_df(
        df_prob_pct=pos_df2,
        start_col=2,
        min_prob_pct=MIN_PROB,
        max_odd=MAX_ODD
    )

    styled_odds = style_odds_df(
        df_odds=odds_df,
        odd_cols=odd_cols,
        cmap=FH_DARK,
        vmin_log=np.log(1.2),
        vmax_log=np.log(50),
        decimals=2
    )

    tab1, tab2 = st.tabs(['Probabilities', 'Odds'])
    height = 450
    with tab1:
        cols = pos_df2.columns[2:]
        st.dataframe(pos_df2.style.background_gradient(cmap=FH_DARK, gmap=pos_df2[cols], subset=pd.IndexSlice[:, cols], vmin=0, 
                                                    vmax=100, low=0.005, high=0.3, axis=None).format(precision=2), height=height)
    with tab2:
        st.dataframe(styled_odds, use_container_width=True, height=height)

    df_probs = (pos_df / 100.0).copy()
    df_probs.columns = list(range(1, df_probs.shape[1] + 1))  # 1..N
    
    current_pos = (
        current_points.set_index("team_name")
        .loc[teams, "pos"]
        .astype(int)
    )
    
    metrics = table_metrics_from_df_probs(df_probs, current_positions=current_pos)
    
    st.divider()
    
    st.subheader("Monte Carlo Diagnostics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Table Stability", f"{metrics['table_stability_index']:.2f} %", border=True)
    c2.metric("Avg Position Shift", f"{metrics['avg_position_shift']:.3f}", border=True)
    c3.metric("Rank Volatility (SD)", f"{metrics['rank_volatility_sd']:.3f}", border=True)
    c4.metric("Entropy (0–1)", f"{metrics['entropy_norm']:.3f}", border=True)
    
    st.info("""
    **Monte Carlo Table Diagnostics**
    
    • **Table Stability** – Average probability that teams finish exactly in their current position. Higher values indicate a more stable table.
    
    • **Avg Position Shift** – Expected number of positions a team is likely to move from its current rank. Higher values suggest more movement in the table.
    
    • **Rank Volatility (SD)** – Standard deviation of the simulated final positions. It measures how uncertain each team's final ranking is.
    
    • **Entropy (0–1)** – Overall uncertainty of the position distribution. Values closer to 1 indicate a wider range of possible outcomes.
    """)
    
    per_team_df = metrics["per_team"]
    per_team_df = per_team_df.rename(columns={'current_pos': 'Current Pos', 'p_same_pos': 'Table Stability', 'eps_shift': 'Avg Pos Shift', 
                                              'rank_sd': 'Rank Volatility', 'entropy_norm': 'Entropy', 'expected_pos': 'xPos'})
    st.dataframe(per_team_df)
    st.divider()
    st.markdown("""
    ### How the simulation works

    Match outcome probabilities are estimated using the model and the remaining fixtures are simulated thousands of times using Monte Carlo methods.  
    Each simulation produces a possible final league table. The probabilities shown above represent the frequency of each outcome across all simulations.

    ### Interpreting the results

    These probabilities describe the **range of plausible outcomes**, not a single prediction.  
    For example, a 30% probability of finishing 1st means that in roughly 30% of the simulated seasons the team finished at the top of the table.

    ### Important limitations

    The simulation is based on statistical expectations derived from historical data and model estimates.  
    Unexpected events such as injuries, tactical changes, or unusual match dynamics are not explicitly modeled.

    Therefore, results should be interpreted as **probabilistic forecasts rather than guarantees**.
    """)
else:
    st.warning("You must login to access this page.")
    st.stop()

