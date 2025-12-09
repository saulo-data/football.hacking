import streamlit as st

st.set_page_config(page_icon='static/image.png')

pg = st.navigation([
    st.Page('pages/soccer_stats_app.py', title='True Performance'),
    st.Page('pages/leagues_overview.py', title='Leagues Overview'), 
    st.Page('pages/match_analysis.py', title='Matches xG Stats'), 
    st.Page('pages/performance.py', title='Relative Performance'), 
    st.Page('pages/xt_stats_app.py', title='Passes Metrics')
], position='top')

with st.sidebar:
    st.image('static/image.png', 
             caption="Football Hacking Transforms Raw Football Data Into Winning Tactical Insights")
    st.write("Football Hacking is built on a simple principle: data should solve real problems. We analyze the game with depth, purpose, and curiosity, turning raw information into insights that matter. Instead of producing charts for the sake of it, we focus on answering meaningful questions that help coaches, analysts, and decision-makers act with clarity. Our work reflects a commitment to understanding football beyond the surface and to continuously learning from the game itself. Fell free to contact us: footbal.data@saulofaria.com.br")

    st.subheader("Useful Links (PT-BR)")
    st.link_button('Website', 'https://www.footballhacking.com', width='stretch')
    st.link_button("Instagram", "https://www.instagram.com/football.hacking/", width='stretch')
    st.link_button("X", "https://x.com/footballhacking", use_container_width=True)
    st.link_button('YouTube', 'https://www.youtube.com/channel/UCkSw2eyetrr8TByFis0Uyug', width='stretch')

pg.run()
