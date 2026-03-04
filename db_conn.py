from pymongo import MongoClient
import streamlit as st

client = MongoClient(st.secrets['url_con'])
db = client.football_data

client2 = MongoClient(st.secrets['url_board'])
db_local = client2.football_data
col_users = db_local.users
