import streamlit as st
import httpx
from selectolax.parser import HTMLParser
import pandas as pd
from functools import reduce
import numpy as np
from mplsoccer import Radar, FontManager

leagues = {
    "Brazil - Brasileirão A": "brazil",
    "Brazil - Brasileirão B": "brazil2",
    "Brazil - Brasileirão C": "brazil3",         
    "England - Premier League": "england",
    "England - Championship": "england2",
    "England - League One": "england3", 
    "England - League Two": "england4",
    "England - National League": "england5",
    "Germany - Bundesliga": "germany",
    "Germany - 2. Bundesliga": "germany2",
    "Germany - 3. Liga": "germany3",
    "Italy - Serie A": "italy",
    "Italy - Serie B": "italy2", 
    "France - Ligue 1": "france",
    "France - Ligue 2": "france2",
    "Spain - La Liga": "spain",
    "Spain - La Liga 2": "spain2",
    "Portugal - Liga Portugal": 'portugal',
    "Sweden - Allvenskan": "sweden", 
    "Turkey - Super Lig": "turkey",
    "Japan - J1 League": "japan", 
    "Norway - Eliteserien": "norway", 
    "USA - Major League Soccer": "usa",
    "Belgium - Pro League": "belgium",
    "Switzerland - Super League": "switzerland",
    "Netherlands - Eredivisie": "netherlands", 
    "Poland - Ekstraklasa": "poland", 
    "Scotland - Premiership": "scotland", 
    "CzechRepublic - 1. Liga": "czechrepublic", 
    "Greece - Super League": "greece",
    "Austria - Bundesliga": "austria", 
    "Australia - A-League": "australia",
    "Argentina - Liga Profesional": "argentina",
    "Chile - Primera Division": "chile", 
    "Colombia - Primera A - Apertura": "colombia", 
    "Croatia - 1. HNL": "croatia", 
    "Denmark - Superligaen": "denmark",
    "South Korea - K League 1": "southkorea",
}




def get_html(url: str, client: httpx.Client) -> HTMLParser:
    resp = client.get(url)
    html = HTMLParser(resp.text)

    return html


def get_rel_perf(html: HTMLParser) -> pd.DataFrame:
    squads = []
    rpis = []
    trs = html.css(
        ".twelve > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2) > table:nth-child(2) > tbody:nth-child(1) > tr:nth-child(2) > td:nth-child(1) > table:nth-child(1) > tbody:nth-child(2) tr.odd")
    for tr in trs:
        squads_table = tr.css('td:nth-child(2) > a:nth-child(1)')
        rpi_metrics = tr.css('td:nth-child(8) > font:nth-child(1) > b:nth-child(1)')
        for squad, rpi in zip(squads_table, rpi_metrics):
            squads.append(squad.text())
            rpis.append(float(rpi.text()))
    df = pd.DataFrame({
        "Squad": squads,
        "RPI": rpis
    })

    return df


def get_rel_form(html: HTMLParser) -> pd.DataFrame:
    squads = []
    last_8_ppgs = []
    ppgs = []
    trs = html.css(
        ".twelve > table:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2) > table:nth-child(2) > tbody:nth-child(1) > tr:nth-child(2) > td:nth-child(1) > table:nth-child(1) > tbody:nth-child(1) tr.odd")
    for tr in trs:
        squads_table = tr.css('td:nth-child(2) > a:nth-child(1)')
        last_8s = tr.css('td:nth-child(5) > font:nth-child(1)')
        ppgs_table = tr.css('td:nth-child(6)')
        for squad, last_8, ppg in zip(squads_table, last_8s, ppgs_table):
            squads.append(squad.text())
            last_8_ppgs.append(float(last_8.text()))
            ppgs.append(float(ppg.text()))

    df = pd.DataFrame({
        "Squad": squads,
        "Last 8 PPG": last_8_ppgs,
        "PPG": ppgs
    })

    return df


def get_leading_times(html: HTMLParser) -> pd.DataFrame:
    squads = []
    leadings = []
    trs = html.css("div.tab:nth-child(3) > table:nth-child(2) > tbody:nth-child(1) tr.odd")
    for tr in trs:
        squads_table = tr.css('td:nth-child(1)')
        leading_times = tr.css('td:nth-child(4)')
        for squad, leading in zip(squads_table, leading_times):
            squads.append(squad.css_first('a:nth-child(1)').text())
            leadings.append(float(leading.text()[:-1]))
    df = pd.DataFrame({
        "Squad": squads,
        "Leading Time %": leadings
    })

    return df


def get_goals_avg(html: HTMLParser) -> pd.DataFrame:
    squads = []
    avg_for = []
    avg_conc = []
    trs = html.css("#btable > tbody tr.odd")
    for tr in trs:
        squads_table = tr.css('td:nth-child(1) > a:nth-child(1)')
        goals_for = tr.css('td:nth-child(3) > font > b')
        goals_conc = tr.css('td:nth-child(4) > font > b')
        for squad, for_goals, conceded_goals in zip(squads_table, goals_for, goals_conc):
            squads.append(squad.text())
            avg_for.append(float(for_goals.text()))
            avg_conc.append(float(conceded_goals.text()))
    df = pd.DataFrame({
        "Squad": squads,
        "Goals Scored Avg": avg_for,
        "Goals Conceded Avg": avg_conc
    })

    return df

@st.cache_data(show_spinner=False)
def create_df(league: str):

    url1 = f"https://www.soccerstats.com/table.asp?league={league}&tid=rp"
    url2 = f"https://www.soccerstats.com/table.asp?league={league}&tid=re"
    url3 = f"https://www.soccerstats.com/table.asp?league={league}&tid=t"
    url4 = f"https://www.soccerstats.com/table.asp?league={league}&tid=d"

    with httpx.Client() as client:
        trs_1 = get_html(url1, client=client)
        df_1 = get_rel_perf(trs_1)

        trs_2 = get_html(url2, client=client)
        df_2 = get_rel_form(trs_2)

        trs_3 = get_html(url3, client=client)
        df_3 = get_leading_times(trs_3)

        trs_4 = get_html(url4, client=client)
        df_4 = get_goals_avg(trs_4)

        dfs = [df_1, df_2, df_3, df_4]

        df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Squad'],
                                                        how='inner'), dfs)
    return df_merged

def input_form():
    with st.form(key="true_perf"):                 
        if 'home_team' not in st.session_state:
            st.session_state['home_team'] = squads[0]
        
        if 'away_team' not in st.session_state:
            st.session_state['away_team'] = squads[1]
        
        home = st.selectbox(label='Select a Home Team', options=squads, index=0)
        away = st.selectbox(label='Select an Away Team',
                            options=squads, index=1)
                
        
        submitted = st.form_submit_button("Submit")

        return submitted



FONT_URL = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
        'RobotoSlab%5Bwght%5D.ttf')
robotto_bold = FontManager(FONT_URL)

st.set_page_config(page_title="Compare Teams")    

try:
    st.html("""
        <style>
            .stMainBlockContainer {
                max-width:60rem;
            
        }            
        </style>
        """
    )
    
    st.title("Key Stats by League")
    
    league_of_choice = st.selectbox(label='Select a League', options=leagues.keys())
    league = leagues[league_of_choice]
    df = create_df(league=league)
    df_styled = df.style.background_gradient(subset=['RPI', 'Last 8 PPG', 'PPG', 'Leading Time %', 'Goals Scored Avg'],
                                             cmap='Greens').format(precision=2).background_gradient(subset=['Goals Conceded Avg'],
                                                                                                   cmap='RdYlGn_r').format(precision=2)
    st.header(league_of_choice)
    st.dataframe(df_styled, hide_index=True, width=8000, use_container_width=True)
    st.caption("*RPI: Relative Performance Index - Team PPG x Opponents PPG")
    
    st.divider()
    
    st.header("Compare The Teams")
    
    squads = list(df['Squad'])    
    


    

    submitted = input_form()
        
    if submitted:
        st.session_state['home_team'] = home
        st.session_state['away_team'] = away            
        params = list(df.columns[1:])
        low = []
        high = []
                
        for param in params:
            low.append(df[param].min() * 0.5)
            high.append(df[param].max() * 1.1)
                
            lower_is_better = ['Goals Conceded Avg']
                
            radar = Radar(params, low, high,
                            lower_is_better=lower_is_better,
                            round_int=[False] * len(params),
                            num_rings=4,
                            ring_width=1, center_circle_radius=1)
                
            values_1 = df[df['Squad'] == st.session_state['home_team']].select_dtypes(include=np.number).values.flatten().tolist()
            values_2 = df[df['Squad'] == st.session_state['away_team']].select_dtypes(include=np.number).values.flatten().tolist()
                
                
            home_total_avg = sum(values_1[-2:])
            away_total_avg = sum(values_2[-2:])
                
            total_avg = np.round((home_total_avg + away_total_avg) / 2, 2)
                
            fig, ax = radar.setup_axis()  # format axis as a radar
            rings_inner = radar.draw_circles(ax=ax, facecolor='#eaeded', edgecolor='#a6a4a1')  # draw circles
            radar_output = radar.draw_radar_compare(values_1, values_2, ax=ax,
                                                            kwargs_radar={'facecolor': '#104DB0', 'alpha': 0.4},
                                                            kwargs_compare={'facecolor': '#B02D10', 'alpha': 0.4})
            radar_poly, radar_poly2, vertices1, vertices2 = radar_output
                
            ax.scatter(vertices1[:, 0], vertices1[:, 1],
                           c='#104DB0', edgecolors='k', marker='o', s=110, zorder=2, alpha=0.5)
            ax.scatter(vertices2[:, 0], vertices2[:, 1],
                        c='#B02D10', edgecolors='k', marker='o', s=110, zorder=2, alpha=0.5)
                
            range_labels = radar.draw_range_labels(ax=ax, fontsize=13, fontproperties=robotto_bold.prop)  # draw the range labels
            param_labels = radar.draw_param_labels(ax=ax, fontsize=18, fontproperties=robotto_bold.prop)
            lines = radar.spoke(ax=ax, color='#a6a4a1', linestyle='--', zorder=2, alpha=0.3)
                
            col1, col2, col3 = st.columns(3, gap='medium', border=True)
                
            with col1:
                st.markdown(f'<span style="font-size: 24px; color: #104DB0; font-weight: bold;">{home}</span>', unsafe_allow_html=True)
                
            with col2:
                st.metric(label="Goals Avg", value=total_avg)
                
            with col3:
                st.markdown(f'<span style="font-size: 24px; color: #B02D10; font-weight:bold;">{away}</span>', unsafe_allow_html=True)
                
                
        st.pyplot(fig)
        st.caption("*Goals Conceded Avg is showed in a lower-is-better mode. i.e: the larger is the area, "
                       "the lower is the average of goals conceded.")
except Exception:
    st.title(
            "Ops! Something Went Wrong\nMaybe this league hasn't started yet. Come back later."
        )
