#libraries
from pymongo import collection
from football_main_app import col_fotmob
import pandas as pd
import numpy as np
import plotly.express as px
from statistics import mean
import streamlit as st

col = col_fotmob

cups = ['INT', 'INT-2']
YEAR = 2026
SEASONS = [f"{YEAR}", f"{YEAR-1}/{YEAR}"]

#get data from mongodb database
@st.cache_data(show_spinner=False)
def get_stats(cups: list, team: str, league: str, seasons: list) -> list:
    stats = list(col.aggregate([{"$match": {"general.country": {"$nin": cups}, 'general.season': {'$in': seasons}, "general.league": league, "$or": [{"teams.home.name": team}, {"teams.away.name": team}]}}, 
                       {"$project": {"_id": 0, "general.round": 1, 'general.season': 1, "general.league": 1,"teams.home.name": 1, "teams.away.name": 1, "stats": 1, 'result': 1}}]))
    return stats
    

def get_teams_dict(venue: str, collection: collection, seasons: list) -> dict:
    teams_data = {}
    teams = list(collection.find({'general.country': {"$nin": cups}, 'general.season': {'$in': seasons}}, {"general.country": 1, "general.league": 1, 'general.season': 1, f"teams.{venue}.name": 1}))

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

#get percentage
def get_perc(x: float, y: float) -> float:
    perc = np.round(((x / y) - 1) * 100, 2)

    if perc == 1:
        perc = 0.0
    
    return perc

#create dataframe from data obtained from mongodb
def get_dataframe(squad_stats: list, team:str, seasons: list) -> pd.DataFrame:
    matchweeks = []
    venues = []
    opps = []
    bp_diff = []
    pass_op_diff = []
    xg_op_diff = []
    touch_opp_diff = []
    results = []

    

    for stat in squad_stats:
        matchweek = int(stat['general']['round'][6:]) if len(stat['general']['round']) > 6 else int(stat['general']['round'])
        
        if matchweek > 5:
            
            home = stat['teams']['home']['name']
            away = stat['teams']['away']['name']
        
            opp = away if home == team else home
    
            venue = 'home' if home == team else 'away'
            venue_opp = 'away' if home == team else 'home'
                
            bp = stat['stats']['ball_possession'][venue_opp]
            pass_op = stat['stats']['passes_opp_half_%'][venue_opp]
            xg_op = stat['stats']['xg_op_for_100_passes'][venue_opp]
            touch_opp = stat['stats']['touch_opp_box_100_passes'][venue_opp]
            

            if stat['result'] == venue:
                result = 'Win'
            elif stat['result'] == venue_opp:
                result = 'Loss'
            else:
                result = 'Draw'

            opps.append(opp)
            matchweeks.append(matchweek)
            venues.append(venue.title())
            results.append(result)
            
            bp_opp = []
            pass_op_opp = []
            xg_op_opp = []
            touch_opp_opp = []

            opp_stats = get_stats(cups=cups, team=opp, league=stat['general']['league'], seasons=seasons)
            for stat_opp in opp_stats:                    
                matchweek_opp = int(stat_opp['general']['round'][6:]) if len(stat_opp['general']['round']) > 6 else int(stat_opp['general']['round'])
                if matchweek_opp < matchweek:
                    home = stat_opp['teams']['home']['name']
                    venue = 'home' if home == opp else 'away'
    
                    bp_opp.append(stat_opp['stats']['ball_possession'][venue])
                    pass_op_opp.append(stat_opp['stats']['passes_opp_half_%'][venue])
                    xg_op_opp.append(stat_opp['stats']['xg_op_for_100_passes'][venue])
                    touch_opp_opp.append(stat_opp['stats']['touch_opp_box_100_passes'][venue])
                else:
                    continue

            bp_diff.append(get_perc(bp, mean(bp_opp)))
            pass_op_diff.append(get_perc(pass_op, mean(pass_op_opp)))
            xg_op_diff.append(get_perc(xg_op, mean(xg_op_opp)))
            touch_opp_diff.append(get_perc(touch_opp, mean(touch_opp_opp)))

    df = pd.DataFrame({
        "Matchweek": matchweeks, 
        "Venue": venues, 
        "Result": results,
        "Opponent": opps,
        "Ball Poss Diff %": bp_diff, 
        "Pass Opp Half Diff %": pass_op_diff, 
        "Open-Play xG per 100 Passes Diff %": xg_op_diff, 
        "Touch Opp Box 100 Passes Diff %": touch_opp_diff
    }) 

    df['Overall Diff %'] = df.iloc[:, 4:].mean(axis=1)
    df['Weighted Avg Diff %'] = (df['Ball Poss Diff %'] * 0.12) + (df['Pass Opp Half Diff %'] * 0.25) + (df['Open-Play xG per 100 Passes Diff %'] * 0.4) + (df['Touch Opp Box 100 Passes Diff %']) * 0.32) / (0.12 + 0.25 + 0.4 + 0.32)
    df['Standard Dev'] = df.iloc[:, 4:8].std(axis=1)
     
    return df.sort_values(by='Matchweek')

st.set_page_config(
    page_title='Squad Report', 
    layout='wide', 

)


st.title("Squad Analysis Based on Relative Performance (from round 5 on) - Current Season")
st.subheader("How much does the opponents’ average performance decrease when facing the selected squad?")
st.write("Except for Standard Deviation, all metrics are presented so that lower values indicate better performance.")

squads = get_teams_dict(venue='home', collection=col, seasons=SEASONS)

try:
            # squads_list = sorted(squads.keys())
            squad = st.selectbox(label='Select a Squad', options=sorted(squads.keys()), index=0)
            squad_data = col.find_one({'general.country': squads[squad]['country'], 'teams.home.name': squads[squad]['name']})
   
            stats = get_stats(cups=cups, team=squads[squad]['name'], league=squads[squad]['league'], seasons=SEASONS)
            df = get_dataframe(stats, team=squads[squad]['name'], seasons=SEASONS)
            

            df_styled = df.style.background_gradient(cmap='RdYlGn_r', text_color_threshold=0.5, 
                                                        subset=df.columns[4:10], low=0.00).background_gradient(cmap='Greens_r', 
                                                                                                                text_color_threshold=0.5, 
                                                                                                                subset=df.columns[-1:], vmin=0).format(precision=2)

            st.divider()
            
            col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment='center')

            with col1:
                st.image(f"{squad_data['teams']['home']['image']}")

            with col2:
                st.metric(label="Last 8 xG Open Play for 100 Passes Diff %", value=np.round(df.tail(8)['Open-Play xG per 100 Passes Diff %'].mean(), 2))

            with col3:
                st.metric(label="Last 8 Pass Opp Half Diff %", value=np.round(df.tail(8)['Pass Opp Half Diff %'].mean(), 2))

            with col4:
                st.metric(label="Last 8 Weighted Avg Diff %", value=np.round(df.tail(8)['Weighted Avg Diff %'].mean(), 2))

            with col5:
                st.metric(label="Last 8 Standard Deviation", value=np.round(df.tail(8)['Standard Dev'].mean(), 2))

            st.divider()
                

                
            st.dataframe(df_styled, hide_index=True)
                
            st.divider()

            st.subheader(f"{squad} TreeMap")
            st.write('The size of the opponents’ squares is determined by the Standard Deviation')

            fig_tree = px.treemap(data_frame=df, path=[px.Constant(squad_data['general']['league']), 'Result', 'Venue', 'Opponent'], values='Standard Dev', 
                                      color='Weighted Avg Diff %', color_continuous_scale='RdYlGn_r')
            fig_tree.update_traces(marker=dict(cornerradius=5))
            fig_tree.update_layout(margin = dict(t=5, l=5, r=1, b=5))
            st.plotly_chart(fig_tree, theme='streamlit')

            st.divider()

            col6, col7 = st.columns(2)

                #chart1
            with col6:
                fig = px.box(data_frame=df, x='Venue', y='Weighted Avg Diff %', color='Venue',
                                color_discrete_sequence=['#104DB0', '#B02D10'], category_orders={'Venue': ['Home', 'Away']})
                    
                st.plotly_chart(fig, theme='streamlit')

                #chart2
            with col7:
                fig2 = px.scatter(data_frame=df, x='Standard Dev', y='Weighted Avg Diff %', color='Matchweek', opacity=0.75, 
                                    color_continuous_scale='greens', hover_name='Opponent')
                fig2.update_traces(marker=dict(size=20,
                                    line=dict(width=2,
                                                color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
                fig2.add_vline(x=15)
                fig2.add_hline(y=0)
                fig2.update_yaxes(tick0=-100, dtick=30)
                fig2.update_xaxes(tick0=0, dtick=10)               


                st.plotly_chart(fig2, theme='streamlit')                


except Exception as e:
    st.text(e)
    st.write("Ops! Something Went Wrong! - Maybe You've Chosen a League which Hasn't Started Yet Or Has Less Than 6 Matchweeks.")
    
st.caption("Created by Saulo Faria - Data Scientist Specialized in Football")
           
