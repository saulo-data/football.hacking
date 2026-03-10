import streamlit as st
from pymongo import MongoClient
from datetime import datetime
from db_conn import db_local

st.set_page_config(initial_sidebar_state="expanded", page_icon='static/image.png')
col_users = db_local.users

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

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
    try:
        emails = col_users.distinct('email')
        
        user = {
                'given_name': st.user.given_name, 
                'family_name': st.user.family_name, 
                'name': st.user.name, 
                'email': st.user.email
            }
            
        if 'user' not in st.session_state:
            st.session_state['user'] = user
    
       
        pg_2 = st.navigation([
            st.Page('pages/home.py', title='Home'),
            st.Page('pages/poisson_preds.py', title='Poisson Preds'),
            st.Page('pages/tables_preds.py', title='Tables Preds'),
            st.Page('pages/about.py', title='About Me')
        ], position='top')
    
        if st.session_state['user']['email'] in emails:
            user_on_db = list(col_users.find({'email': st.session_state['user']['email']}))[0]
            st.session_state['user']['plan'] = user_on_db['plan']
            col_users.update_one({'email': st.session_state['user']['email']}, {"$inc": {'number_of_access': 1}, "$set": {'last_seen_on': datetime.now()}})
            
        else:
            st.session_state['user']['number_of_access'] = 1
            st.session_state['user']['last_seen_on'] = datetime.now()
            st.session_state['user']['on_mailing_list'] = False
            st.session_state['user']['plan'] = 'free'
            col_users.insert_one(st.session_state['user'])

        st.session_state['logged_in'] = True
        pg_2.run()
        
        if st.button("Log out", width=250, type='primary'):
            st.logout()
        st.write(f"Hello, {st.session_state['user']['name']}!")
        if st.session_state['user']['plan'] == 'premium':
            st.badge("Plan: Premium", icon=":material/star_shine:", color="green")
        elif st.session_state['user']['plan'] == 'free':
            st.badge("Plan: Free")
            st.warning("Become a premium subscriber to the Football Hacking newsletter on Substack to get access to all the leagues in our database and receive exclusive content straight to your inbox. Link in the sidebar (Website).")
        else:
            st.warning('Something went wrong!')
    except Exception as e:
        st.text(e)
        

with st.sidebar:
    st.image('static/image.png', 
             caption="Football Hacking Transforms Raw Football Data Into Winning Tactical Insights")
    st.write("Football Hacking is built on a simple principle: data should solve real problems. We analyze the game with depth, purpose, and curiosity, turning raw information into insights that matter. Instead of producing charts for the sake of it, we focus on answering meaningful questions that help coaches, analysts, and decision-makers act with clarity. Our work reflects a commitment to understanding football beyond the surface and to continuously learning from the game itself. Fell free to contact us: footbal.data@saulofaria.com.br")

    st.subheader("My Links")
    st.link_button("Premium Dataset (82,000 matches)", "https://saulofaria0.gumroad.com/l/football-matches-dataset", width='stretch')
    st.link_button('Website', 'https://www.footballhacking.com', width='stretch')
    st.link_button('E-Book MplSoccer', 'subscribepage.io/mplsoccer', width='stretch')
    st.link_button('E-Book Aposta Consciente', "subscribepage.io/aposta-consciente", width='stretch')
    st.link_button("Instagram", "https://www.instagram.com/football.hacking/", width='stretch')
    st.link_button("X", "https://x.com/footballhacking", use_container_width=True)
    st.link_button('YouTube', 'https://www.youtube.com/channel/UCkSw2eyetrr8TByFis0Uyug', width='stretch')
    st.link_button('LinkedIn', 'https://www.linkedin.com/in/saulo-faria-data/', width='stretch')
