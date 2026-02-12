#libraries
import streamlit as st
import pandas as pd
from pymongo import collection
import plotly.express as px
from football_main_app import col_fotmob


col = col_fotmob

#list of the teams
cups = ['INT', 'INT-2']
YEAR = 2026
SEASONS = [f"{YEAR}", f"{YEAR-1}/{YEAR}"]

leagues = col.find({'general.country': {"$nin": cups}, 'general.season': {'$in': SEASONS}}).distinct('general.league')
away_teams = col.distinct('teams.away.name')
color_home = '#104DB0'
color_away = '#B02D10'

#function to get the data of the selected match
@st.cache_data(show_spinner=False)
def get_match(home: str, away: str, seasons: list) -> dict:
    match = col.find_one({'teams.home.name': home, 'teams.away.name': away, 'general.season': {'$in': seasons}})

    return match

#function to get complete names for teams
def get_teams_dict(venue: str, collection: collection, exclude: list, seasons: list) -> dict:
    teams_data = {}
    teams = list(collection.find({'general.country': {"$nin": exclude}, 'general.season': {'$in': seasons}}, {"general.country": 1, "general.league": 1, f"teams.{venue}.name": 1}))
    

    for team in teams:
        team_name = team['teams'][venue]['name']
        team_league = team['general']['league']
        team_country = team['general']['country']
        complete_name = f"{team_name} - {team_country}"
        if complete_name not in teams_data.keys():
            teams_data[complete_name] = {'country': team_country, 'league': team_league, 'name': team_name}
        else:
            continue
    
    return teams_data

def categorize_shot(shot):
        if shot >= 0.7:
            return 'High'
        elif 0.7 > shot >= 0.3:
            return 'Medium'
        else:
            return 'Low'
            
def input_form():
    with st.form(key='match_stats'):
        if 'home' not in st.session_state:
            st.session_state['home'] = home_teams[home_names[0]]['name']
        
        if 'away' not in st.session_state:
            st.session_state['away'] = away_teams[away_names[0]]['name']
    
        col1, col2 = st.columns(2)
    
        with col1:
            home = st.selectbox(label="Select a Home Team", options=home_names, index=0)
        
        with col2:
            away = st.selectbox(label="Select an Away Team", options=away_names, index=1)
    
        submitted = st.form_submit_button("Submit")

    return submitted, home, away

st.set_page_config(page_title='Match Plots', layout='wide')

#page title
st.header('Plot the stats of a selected match – national leagues only')
home_teams = get_teams_dict(venue='home', collection=col, exclude=cups, seasons=SEASONS)
away_teams = get_teams_dict(venue='away', collection=col, exclude=cups, seasons=SEASONS)
home_names = list(home_teams.keys())
away_names = list(away_teams.keys())

#form to select the teams
submitted, home, away = input_form()

if submitted:
    try:
        st.session_state['home'] = home_teams[home]['name']
        st.session_state['away'] = away_teams[away]['name']

        match = get_match(st.session_state['home'], st.session_state['away'], SEASONS)
        if not match:
            st.text("This match may not have occurred yet, or the teams may not belong to the same national league")
        else:
            scoreline = f"{match['score']['home']} x {match['score']['away']}"
            match_details = f"{match['general']['country']} - {match['general']['league']} - Season {match['general']['season']}"

            stats = match['stats']

            df_stats = pd.DataFrame.from_dict(stats)
            df_stats['team'] = [st.session_state['home'], st.session_state['away']]
            df_stats = df_stats.melt(id_vars=['team'], value_vars=df_stats.columns)

            touch_100 = df_stats[df_stats['variable'] == 'touch_opp_box_100_passes']['value'].values

            df_stats = df_stats[df_stats['variable'] != 'touch_opp_box_100_passes']
            df_stats = df_stats.replace({
                    'ball_possession': 'Ball Poss', 
                    'passes_opp_half_%': 'Passes Opp Half %', 
                    'xg_op_for_100_passes': 'Open-Play xG per 100 Passes', 
                    'interceptions_perc': 'Interceptions %'
                })
                
                
            shots = match['shotmap']
            home_shots = pd.DataFrame.from_dict(shots['home'])
            home_shots['team'] = match['teams']['home']['name']
            away_shots = pd.DataFrame.from_dict(shots['away'])
            away_shots['team'] = match['teams']['away']['name']
            df_shots = pd.concat([home_shots, away_shots]).fillna(0)
            df_shots['size'] = [s + 0.05 for s in df_shots['xgot']]
            df_shots['efficiency_rate'] = (df_shots['xg'] + df_shots['xgot']) / 2

            st.subheader(match_details)
    
            col1, col2, col3 = st.columns([3, 12, 3], vertical_alignment='center', gap='large')

            with col1:
                st.image(f"{match['teams']['home']['image']}", use_container_width=True)
            with col2:
                st.markdown(f"<h1 style='text-align: center;'>{match['teams']['home']['name']} - {scoreline} - {match['teams']['away']['name']}</h1>", unsafe_allow_html=True)
            with col3:
                st.image(f"{match['teams']['away']['image']}", use_container_width=True)

            columns_shown = ['min', 'team', 'player', 'type', 'situation', 'outcome', 'xg', 'xgot', 'efficiency_rate']
            st.dataframe(df_shots[columns_shown].sort_values(by='min'), hide_index=True)

            fig = px.scatter(data_frame=df_shots, x='min', y='xg', color='team', color_discrete_sequence=[color_home, color_away], size='size', symbol='outcome', 
                                    hover_name='team', hover_data={'xgot': True, 'team': False, 'player': True, 'type': True, 'situation': True, 'outcome': True, 'size': False},  
                                    range_y=[0, 1], title='xG by Minute and its xGOT (Size)', labels={'min': 'Minutes', 'xg': 'xG'}
                                    
                                 )
            fig.update_xaxes(tickvals=[0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90])
            fig.update_layout(legend_title_text='Squads')
                
            fig_bar = px.bar(data_frame=df_shots, x='situation', y='xg', color='team', color_discrete_sequence=[color_home, color_away], barmode='group', opacity=0.75, 
                                 title='xG By Situation', labels={'situation': '', 'xg': ''})
            fig_bar.update_layout(showlegend=False)

            fig_polar = px.line_polar(data_frame=df_stats, r='value', theta='variable', line_close=True, color='team', color_discrete_sequence=[color_home, color_away], 
                                          title='Main Stats of the Match')
            fig_polar.update_traces(fill='toself')
            fig_polar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]  # Set the desired min and max values for the radial axis
                            )
                        ),
                        showlegend=False
                )    

            fig_box = px.box(data_frame=df_shots, x='team', y='efficiency_rate', color='team', color_discrete_sequence=[color_home, color_away], 
                                 title='Shot Efficiency – xG/xGOT Mean', labels={'efficiency_rate': '', 'team': ''})
            fig_box.update_layout(showlegend=False)

            st.divider()

            st.plotly_chart(fig)

            st.divider()

            st.plotly_chart(fig_bar)

            st.divider()

            st.subheader("Touches in the Opponent’s Penalty Box per 100 Passes")

            col4, col5 = st.columns(2, vertical_alignment='center', border=True, gap='large')

            with col4:
                st.metric(label=st.session_state['home'], value=touch_100[0])
            with col5:
                st.metric(label=st.session_state['away'], value=touch_100[1])

            st.divider()

            col6, col7 = st.columns(2, vertical_alignment='center')                

            with col6:
                st.plotly_chart(fig_polar)
            with col7:
                st.plotly_chart(fig_box)            
            
    except Exception as e:
        st.text(e)
