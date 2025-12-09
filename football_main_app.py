import streamlit as st
from pymongo import MongoClient
from datetime import datetime

client = MongoClient(st.secrets['url_board'])
db_local = client.football_data
col_users = db_local.users

st.set_page_config(page_icon='static/image.png')

if not st.user.is_logged_in:
    st.text("""Please log in to access the full functionality of the platform. Logging in ensures a personalized experience and allows you to use all available features and analytical tools.""")
    
    pg_1 = st.navigation([
        st.Page('pages/home.py', title='Home'),
        st.Page('pages/about.py', title='About Me')
    ], position='top')

    pg_1.run()
    if st.button("Log in with Google", width=250, type='primary'):
        st.login()
else:
    emails = col_users.distinct('email')
    user = {
            'given_name': st.user.given_name, 
            'family_name': st.user.family_name, 
            'name': st.user.name, 
            'email': st.user.email
        }
    
    pg_2 = st.navigation([
        st.Page('pages/home.py', title='Home'),
        st.Page('pages/soccer_stats_app.py', title='True Performance'),
        st.Page('pages/leagues_overview.py', title='League Overview'), 
        st.Page('pages/match_analysis.py', title='Match xG Stats'), 
        st.Page('pages/performance.py', title='Relative Performance'), 
        st.Page('pages/xt_stats_app.py', title='Passes Metrics'),
        st.Page('pages/about.py', title='About Me')
    ], position='top')

    if user['email'] in emails:
        col_users.update_one({'email': user['email']}, {"$inc": {'number_of_access': 1}, "$set": {'last_seen_on': datetime.now()}})
    else:
        user['number_of_access'] = 1
        user['last_seen_on'] = datetime.now()
        col_users.insert_one(user)

    pg_2.run()
    
    if st.button("Log out", width=250, type='primary'):
        st.logout()
    st.write(f"Hello, {user['name']}!")

with st.sidebar:
    st.image('static/image.png', 
             caption="Football Hacking Transforms Raw Football Data Into Winning Tactical Insights")
    st.write("Football Hacking is built on a simple principle: data should solve real problems. We analyze the game with depth, purpose, and curiosity, turning raw information into insights that matter. Instead of producing charts for the sake of it, we focus on answering meaningful questions that help coaches, analysts, and decision-makers act with clarity. Our work reflects a commitment to understanding football beyond the surface and to continuously learning from the game itself. Fell free to contact us: footbal.data@saulofaria.com.br")

    st.subheader("Useful Links (PT-BR)")
    st.link_button('Website', 'https://www.footballhacking.com', width='stretch')
    st.link_button("Instagram", "https://www.instagram.com/football.hacking/", width='stretch')
    st.link_button("X", "https://x.com/footballhacking", use_container_width=True)
    st.link_button('YouTube', 'https://www.youtube.com/channel/UCkSw2eyetrr8TByFis0Uyug', width='stretch')
