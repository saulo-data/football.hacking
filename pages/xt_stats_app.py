import streamlit as st
import pandas as pd
import numpy as np
from mplsoccer.pitch import Pitch
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from football_main_app import col_whoscored_calendar, col_whoscored_matches
import json

#setting colors
background = '#e1ece1'
text = '#073d05'

#getting xT Data
with open('static/xt_metrics.json', 'r') as f:
    xt_metrics = json.load(f)

xt_array = np.array(xt_metrics)

col_calendar = col_whoscored_calendar
col_matches = col_whoscored_matches

# function to get teams names
@st.cache_data(ttl='12h', show_spinner=False)
def get_names(venue: str) -> list[str]:
    teams = col_calendar.find({'season': {'$in': [2025, 2526]}}).distinct(f'{venue}_team')
    
    return teams

# function to get match id
@st.cache_data(ttl='12h', show_spinner=False)
def get_match_data(home_team: str, away_team: str) -> int:
    match_data = list(col_calendar.find({'home_team': home_team, 'away_team': away_team, 'season': {'$in': [2025, 2526]}}, {'_id': 0, 'game_id': 1, 'league': 1, "home_team": 1, "away_team": 1, 
                                                                                    "home_score": 1, "away_score": 1}))
    return match_data[0]

# function to obtain inital dataframe of passes
@st.cache_data(ttl='12h', show_spinner=False)
def get_passes_df(match_id: int, xt_array: np.array) -> pd.DataFrame:
    xt_rows, xt_cols = xt_array.shape

    corrections = {'Man City': 'Manchester City', 'Man Utd': 'Manchester United', 'Mainz': 'Mainz 05', 
    'Leverkusen': 'Bayer Leverkusen', 'Stuttgart': 'VfB Stuttgart', 'RBL': 'RB Leipzig', 'Bayern': 'Bayern Munich', 
    'Hamburg': 'Hamburger SV', 'Atletico': 'Atletico Madrid', 'Porto': 'FC Porto', 'Sporting': 'Sporting CP', 
    'PSG': 'Paris Saint-Germain', 'Seattle': 'Seattle Sounders FC', 'Vancouver': 'Vancouver Whitecaps'}

    calendar = list(col_calendar.find({'game_id': match_id}, {'_id': 0, 'home_team': 1, 'away_team': 1}))[0]
    home_team = calendar['home_team']
    away_team = calendar['away_team']

    events = list(col_matches.find({'match_id': match_id, 'events.type.displayName': {"$ne": "FreekickTaken"}}, {'events': 1, '_id': 0}))
    columns = ['period', 'minute', 'second', 'type', 'outcome_type', 'team', 'player', 'x', 'y', 'end_x', 'end_y']
    df = pd.DataFrame.from_dict(events[0]['events'])[columns]
    df.loc[:, 'recipient'] = df[['player']].shift(-1)

    succes_df = df[df['outcome_type'] == 'Successful']
    passes_df = succes_df[succes_df['type'] == 'Pass']

    passes_df.loc[:, 'team'] = passes_df.replace(corrections)

    passes_df.loc[:, 'x1_bin'] = pd.cut(passes_df['x'], bins=xt_cols, labels=False)
    passes_df.loc[:, 'y1_bin'] = pd.cut(passes_df['y'], bins=xt_rows, labels=False)
    passes_df.loc[:, 'x2_bin'] = pd.cut(passes_df['end_x'], bins=xt_cols, labels=False)
    passes_df.loc[:, 'y2_bin'] = pd.cut(passes_df['end_y'], bins=xt_rows, labels=False)

    passes_df.loc[:, 'start_zone_xt'] = passes_df[['x1_bin', 'y1_bin']].apply(lambda x: xt_array[x[1]][x[0]], axis=1)
    passes_df.loc[:, 'end_zone_xt'] = passes_df[['x2_bin', 'y2_bin']].apply(lambda x: xt_array[x[1]][x[0]], axis=1)
    passes_df.loc[:, 'xt_final'] = round((passes_df['end_zone_xt'] - passes_df['start_zone_xt']), 2)

    home_passes_df = passes_df[passes_df['team'] == home_team]
    home_passes_df['xt_cumulative'] = home_passes_df['xt_final'].cumsum()

    away_passes_df = passes_df[passes_df['team'] == away_team]
    away_passes_df.loc[:, 'x'] = (away_passes_df['x']-100) * -1
    away_passes_df.loc[:, 'end_x'] = (away_passes_df['end_x']-100) * -1
    away_passes_df.loc[:, 'y'] = (away_passes_df['y']-100) * -1
    away_passes_df.loc[:, 'end_y'] = (away_passes_df['end_y']-100) * -1
    away_passes_df['xt_cumulative'] = away_passes_df['xt_final'].cumsum()

    passes_df = pd.concat([home_passes_df, away_passes_df])

    return passes_df

# function to obtain grouped dataframe of passes
def get_grouped_pass_df(data: pd.DataFrame) -> pd.DataFrame:
    df_grouped = data.groupby(['team', 'player'], as_index=False).agg({'x': ['max', 'mean', 'std'], 'y': ['max', 'mean', 'std'], 'start_zone_xt': 'mean', 'end_zone_xt': 'mean', 'xt_final': 'mean'})
    df_grouped.columns = ['team', 'player', 'x_max', 'x_mean', 'x_std', 'y_max', 'y_mean', 'y_std', 'start_zone_xt_mean', 'end_zone_xt_mean', 'xt_final_mean']

    return df_grouped.reset_index().sort_values(by='player')

# function to make graphs
def make_graphs(data: pd.DataFrame, home_team: str, away_team: str) -> nx.DiGraph:
    combinations_df = data[['team', 'player', 'recipient']]
    combinations_df.loc[:, 'passes'] = 1
    combinations_grouped_df = combinations_df.groupby(['team', 'player', 'recipient']).sum().reset_index()

    home_combinations = combinations_grouped_df[combinations_grouped_df['team'] == home_team]
    away_combinations = combinations_grouped_df[combinations_grouped_df['team'] == away_team]

    G_home, G_away = nx.DiGraph(), nx.DiGraph()
    G_home.add_nodes_from(home_combinations['player'].unique())
    G_away.add_nodes_from(away_combinations['player'].unique())

    for index, row in combinations_grouped_df.iterrows():
        if row['team'] == home_team:
            nodes = G_home.nodes
            graph = G_home
        elif row['team'] == away_team:
            nodes = G_away.nodes
            graph = G_away
        else:
            pass

        if row['player'] and row['recipient'] in nodes:
            graph.add_edge(row['player'], row['recipient'], passes=row['passes'])
        
    return G_home, G_away

# function to plot graphs
def plot_graphs(graph_home: nx.DiGraph, graph_away: nx.DiGraph, home_color: str, away_color: str) -> None:
    try:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        node_size = 500
        font_size = 10
        font_weight = 'bold'
        width = 2.5
        weights_home= nx.get_edge_attributes(graph_home, 'passes').values()
        weights_away= nx.get_edge_attributes(graph_away, 'passes').values()
        pos_home = nx.kamada_kawai_layout(graph_home)
        pos_away = nx.kamada_kawai_layout(graph_away)
        

        nx.draw(G=graph_home, with_labels=True, node_size=node_size, node_color=home_color, pos=pos_home, font_size=font_size, font_weight=font_weight, edge_color=weights_home, 
                edge_cmap=plt.cm.Greys, width=width, ax=ax[0])
        nx.draw(G=graph_away, with_labels=True, node_size=node_size, node_color=away_color, pos=pos_away, font_size=font_size, font_weight=font_weight, edge_color=weights_away, 
                edge_cmap=plt.cm.Greys, width=width, ax=ax[1])
        fig.patch.set_facecolor(background)

        st.subheader('Players Interactions')
        st.pyplot(fig)
    except Exception:
        st.text('There is not enough data available.')


# function to get graphs of a team
def get_centralities_df(data: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    
    G_home, G_away = make_graphs(data, home_team, away_team)

    home_centralities = pd.DataFrame({
        'player': nx.load_centrality(G_home).keys(), 
        'load centrality': nx.load_centrality(G_home).values(),
        'closeness centrality': nx.closeness_centrality(G_home).values(),
        'eigenvector centrality': nx.eigenvector_centrality(G_home).values(), 
        'pagerank': nx.pagerank(G_home).values()
    })

    away_centralities = pd.DataFrame({
        'player': nx.load_centrality(G_away).keys(), 
        'load centrality': nx.load_centrality(G_away).values(),
        'closeness centrality': nx.closeness_centrality(G_away).values(),
        'eigenvector centrality': nx.eigenvector_centrality(G_away).values(), 
        'pagerank': nx.pagerank(G_away).values()
    })

    centralities = pd.concat([home_centralities, away_centralities]).sort_values(by='player')
    
    return centralities

# function to plot cumulative xT
def plot_xt_cumulative(data: pd.DataFrame, home_team: str, away_team: str, home_color: str, away_color: str) -> None:
    home_passes_df = data[data['team'] == home_team]
    away_passes_df = data[data['team'] == away_team]
    home_passes_grouped_min = home_passes_df.groupby('minute')['xt_cumulative'].mean().reset_index()
    away_passes_grouped_min = away_passes_df.groupby('minute')['xt_cumulative'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    # ax.set_title('xT Accumulated Performance By Minute (Only Passes)', fontweight='bold')
    ax.set_xlabel('Minutes')
    ax.set_xticks(range(0, data['minute'].max(), 10))
    ax.set_ylabel('Expected Threat (xT)')  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sns.set_theme('paper')
    sns.lineplot(data=home_passes_grouped_min, x='minute', y='xt_cumulative', ax=ax, color=home_color, alpha=0.65, drawstyle='steps-pre', err_style=None, label=home_team)
    sns.lineplot(data=away_passes_grouped_min, x='minute', y='xt_cumulative', ax=ax, color=away_color, alpha=0.65, drawstyle='steps-pre', err_style=None, label=away_team)
    
    ax.fill_between(x=home_passes_grouped_min['minute'], y1=home_passes_grouped_min['xt_cumulative'], step='pre', alpha=0.1, color=home_color, interpolate=True)
    ax.fill_between(x=away_passes_grouped_min['minute'], y1=away_passes_grouped_min['xt_cumulative'], step='pre', alpha=0.1, color=away_color, interpolate=True)
    fig.patch.set_facecolor(background)
    ax.patch.set_facecolor(background)
    plt.legend(facecolor=background, framealpha=1)
    st.subheader('Team Expected Threat (xT) Accumulation — Passes Only')
    st.text('This chart shows the level of danger a team has generated against its opponent')
    st.pyplot(fig)
    
# function to plot dispersion of the players
@st.fragment
def plot_dispersion_scatter(data: pd.DataFrame, home: str, away: str, home_color: str, away_color: str) -> None:
    st.subheader('Player Positional Dispersion (Standard Deviation) for the Match')

    metrics = data.columns[-4:]

    if 'size' not in st.session_state:
        st.session_state['size'] = metrics[0]
    
    size = st.radio(label="Choose which centrality measure you want to use for circle size", options=metrics, index=0, horizontal=True)
    st.session_state['size'] = size

    fig = px.scatter(data_frame=data, x='x_std', y='y_std', color='team', size=st.session_state['size'], 
        category_orders={'team': [home, away]}, text='player', opacity=0.5, hover_data=['player'], 
        color_discrete_sequence=[home_color, away_color], labels={'x_std': 'Depth Dispersion', 'y_std': 'Breadth Dispersion'})
    fig.update_traces(textfont_weight='bold', textposition='bottom center')
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig)

# function to plot 10 most danger passes
def plot_most_danger_passes(data_home: pd.DataFrame, data_away: pd.DataFrame, home_color: str, away_color: str, limit: int) -> None:
    home_df = data_home.sort_values('end_zone_xt', ascending=False).head(limit)
    away_df = data_away.sort_values('end_zone_xt', ascending=False).head(limit)

    pitch = Pitch(pitch_type='opta',  line_zorder=3, line_color='#c7d5cc', pitch_color='#22312b')
    fig, ax = pitch.draw(nrows=1, ncols=2, figsize=(10, 5), tight_layout=True)
    fig.patch.set_facecolor(background)

    # home
    pitch.arrows(xstart=home_df.x, ystart=home_df.y, xend=home_df.end_x, yend=home_df.end_y, color=home_color, ax=ax[0], 
    label=f"{home} {limit} Most Dangerous Passes", width=1.5, alpha=0.65)

    # away
    pitch.arrows(xstart=away_df.x, ystart=away_df.y, xend=away_df.end_x, yend=away_df.end_y, color=away_color, ax=ax[1], 
    label=f"{away} {limit} Most Dangerous Passes", width=1.5, alpha=0.65)

    st.subheader(f'{limit} Most Dangerous Passes')
    st.text(f'This chart displays the {limit} passes that reached the highest-xT zones in the selected timestamp')
    st.pyplot(fig)

# function to plot players xT along the match
@st.fragment
def plot_xt_player(data: pd.DataFrame, players: list, home_team: str, away_team: str) -> None:
    players_desc = {}
    for index, row in data.iterrows():
        descp = f"{row['player']} - {row['team']}"
        if descp not in players_desc.keys():
            if row['team'] == home_team:
                color = 'blue'
            elif row['team'] == away_team:
                color = 'red'
            players_desc[descp] = {'player': row['player'], 'color': color}
    
    if 'player_xt' not in st.session_state:
        st.session_state['player_xt'] = players[0]
    if 'team_color' not in st.session_state:
        st.session_state['team_color'] = 'blue'
    
    st.subheader('Expected Threat (xT) by Player Along 90 min')
    st.text('Player xT Performance Analysis')
    player = st.selectbox(label='Select a Player', options=sorted(players_desc.keys()), index=0)
    st.session_state['player_xt'] = players_desc[player]['player']
    st.session_state['team_color'] = players_desc[player]['color']
    player_xt_df = data[data['player'] == st.session_state['player_xt']]

    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_title(f"{st.session_state['player_xt']}", fontweight='bold')
    ax.set_xlabel('Minutes')
    ax.set_xticks(range(player_xt_df['minute'].min(), player_xt_df['minute'].max(), 5))
    ax.set_ylabel('Expected Threat (xT)')
    sns.set_theme('paper')
    sns.scatterplot(data=player_xt_df, x='minute', y='xt_final', color=st.session_state['team_color'], s=380, alpha=0.6, ax=ax)
    fig.patch.set_facecolor(background)
    ax.patch.set_facecolor(background)

    
    st.pyplot(fig)

# function to plot passes by player
@st.fragment
def plot_passes_by_player(data: pd.DataFrame, players: list, home_team: str, away_team: str) -> None:
    players_desc = {}
    for index, row in data.iterrows():
        descp = f"{row['player']} - {row['team']}"
        if descp not in players_desc.keys():
            if row['team'] == home_team:
                color = 'b'
            elif row['team'] == away_team:
                color = 'r'
            players_desc[descp] = {'player': row['player'], 'color': color}
    
    if 'player_passes' not in st.session_state:
        st.session_state['player_passes'] = players[0]
    if 'team_color' not in st.session_state:
        st.session_state['team_color'] = 'b'
    
    st.subheader('Passes by Player')
    st.text('This plot shows the direction of passes made and received by the player')
    player = st.selectbox(label='Select a Player', options=sorted(players_desc.keys()), index=0, key='passes_player')
    st.session_state['player_passes'] = players_desc[player]['player']
    st.session_state['team_color'] = players_desc[player]['color']
    player_from = data[data['player'] == st.session_state['player_passes']]
    player_to = data[data['recipient'] == st.session_state['player_passes']]

    pitch = Pitch(pitch_type='opta',  line_zorder=3, line_color='#c7d5cc', pitch_color='#22312b')
    fig, ax = pitch.draw(nrows=1, ncols=2, figsize=(10, 5), tight_layout=True)
    ax[0].set_title(f"From {st.session_state['player_passes']}", fontweight='bold')
    ax[1].set_title(f"To {st.session_state['player_passes']}", fontweight='bold')
    fig.patch.set_facecolor(background)

    # From
    pitch.arrows(xstart=player_from.x, ystart=player_from.y, xend=player_from.end_x, yend=player_from.end_y, color=st.session_state['team_color'], ax=ax[0], 
    width=1.5, alpha=0.65)

    # To
    pitch.arrows(xstart=player_to.x, ystart=player_to.y, xend=player_to.end_x, yend=player_to.end_y, color=st.session_state['team_color'], ax=ax[1], 
    width=1.5, alpha=0.65)

    st.pyplot(fig)

# function to plot the passes flow of each team
@st.fragment
def plot_flow_map(data: pd.DataFrame, home_team: str, away_team: str) -> None:
    minutes = data['minute'].unique().tolist()

    arrow_cmap = 'binary'
    arrow_length = 28
    arrow_type = 'scale'

    if 'start' not in st.session_state:
        st.session_state['start'] = minutes[0]
    if 'end' not in st.session_state:
        st.session_state['end'] = minutes[-1]

    st.subheader('Pass Direction, Interactions, and Match Flow Map')
    start, end = st.select_slider(label='Select a Timestamp', options=sorted(minutes), value=(minutes[0], minutes[-1]))
    st.session_state['start'] = start
    st.session_state['end'] = end

    condition_home = data['team'] == home_team
    condition_away = data['team'] == away_team
    condition_start = data['minute'] > st.session_state['start']
    condition_end = data['minute'] <= st.session_state['end']

    home_passes_df = data[condition_home & condition_start & condition_end]
    away_passes_df = data[condition_away & condition_start & condition_end]

    pitch = Pitch(pitch_type='opta',  line_zorder=2, line_color='#c7d5cc', pitch_color='#22312b')
    fig, ax = pitch.draw(nrows=1, ncols=2, figsize=(10, 5), tight_layout=True)
    ax[0].set_title(f"{home} ----> ", fontweight='bold')
    ax[1].set_title(f"<---- {away}", fontweight='bold')
    fig.patch.set_facecolor(background)
    
    bins = (7, 5)

    #home
    bs_heatmap = pitch.bin_statistic(home_passes_df.x, home_passes_df.y, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax[0], cmap='Blues')
    fw = pitch.flow(home_passes_df.x, home_passes_df.y, home_passes_df.end_x, home_passes_df.end_y,
                    cmap=arrow_cmap, arrow_type=arrow_type, arrow_length=arrow_length,
                    bins=bins, zorder=2, ax=ax[0])    

    #away
    bs_heatmap = pitch.bin_statistic(away_passes_df.x, away_passes_df.y, statistic='count', bins=bins)
    hm = pitch.heatmap(bs_heatmap, ax=ax[1], cmap='Reds')

    fw = pitch.flow(away_passes_df.x, away_passes_df.y, away_passes_df.end_x, away_passes_df.end_y,
                    cmap=arrow_cmap, arrow_type=arrow_type,arrow_length=arrow_length,
                    bins=bins, zorder=2, ax=ax[1])

        
    st.pyplot(fig)
    plot_most_danger_passes(data_home=home_passes_df, data_away=away_passes_df, home_color='b', away_color='r', limit=30)
    G_home, G_away = make_graphs(data=pd.concat([home_passes_df, away_passes_df]), home_team=home_team, away_team=away_team)
    plot_graphs(graph_home=G_home, graph_away=G_away, home_color='b', away_color='r')

# set the streamlit page
st.set_page_config(
    page_title='Pass and Expected Threat(xT) Information', 
    layout='wide'
)


# set title of the page
st.title('Pass and Expected Threat(xT) Information - Current Season')

# get teams names
home_names = get_names(venue='home')
away_names = get_names(venue='away')

# form
with st.form(key='xt_app'):
    if 'home' not in st.session_state:
        st.session_state['home'] = home_names[0]
    
    if 'away' not in st.session_state:
        st.session_state['away'] = away_names[0]

    col1, col2 = st.columns(2)

    with col1:
        home = st.selectbox(label="Select a Home Team", options=home_names, index=0)
    
    with col2:
        away = st.selectbox(label="Select an Away Team", options=away_names, index=1)

    submitted = st.form_submit_button("Submit")

    if submitted:
        try:
            st.session_state['home'] = home
            st.session_state['away'] = away

            match = get_match_data(home_team=st.session_state['home'], away_team=st.session_state['away'])
            if not match:
                st.text("This match may not have occurred yet, or the teams may not belong to the same national league")
            else:
                passes_df = get_passes_df(match_id=match['game_id'], xt_array=xt_array)
                passes_grouped_df = get_grouped_pass_df(passes_df)
                centralities_df = get_centralities_df(passes_df, home_team=home, away_team=away)
                passes_grouped_df.loc[:, 'load centrality'] = centralities_df['load centrality'].values
                passes_grouped_df.loc[:, 'closeness centrality'] = centralities_df['closeness centrality'].values
                passes_grouped_df.loc[:, 'eigenvector centrality'] = centralities_df['eigenvector centrality'].values
                passes_grouped_df.loc[:, 'pagerank'] = centralities_df['pagerank'].values
                players = sorted(passes_grouped_df['player'].unique().tolist())


                
                
        except Exception as e:
            st.text('Something is not right! Maybe this match hasn’t occurred yet.')

try:
    col3, col4 = st.columns([6.1, 3.9])
    
    with col3:        
        plot_xt_cumulative(data=passes_df, home_team=st.session_state['home'], away_team=st.session_state['away'], home_color='b', away_color='r')
    
    with col4:
        plot_xt_player(data=passes_df, players=players, home_team=st.session_state['home'], away_team=st.session_state['away'])
             
    st.divider() 
    plot_flow_map(data=passes_df, home_team=st.session_state['home'], away_team=st.session_state['away'])
    st.divider()
    plot_passes_by_player(data=passes_df, players=players, home_team=st.session_state['home'], away_team=st.session_state['away'])
    
    st.divider()
    
    st.subheader('Highest Load Centrality')
    st.text('The lower this value is, the more evenly the team distributes its passes')
    col5, col6 = st.columns(2, gap='large', vertical_alignment='center', border=True)    
    with col5:
        st.metric(label=f"{st.session_state['home']} Highest Load Centrality", 
        value=np.round(passes_grouped_df[passes_grouped_df['team'] == st.session_state['home']]['load centrality'].max(), 2))
    with col6:
        st.metric(label=f"{st.session_state['away']} Highest Load Centrality", 
        value=np.round(passes_grouped_df[passes_grouped_df['team'] == st.session_state['away']]['load centrality'].max(), 2))
    st.divider()
    plot_dispersion_scatter(passes_grouped_df, home_color='blue', away_color='red', home=st.session_state['home'], 
            away=st.session_state['away'])
    
    
    
except Exception as e:
    st.text(' ')


st.caption("Created by Saulo Faria - Data Scientist Specialized in Football")
