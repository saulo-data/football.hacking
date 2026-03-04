from pymongo import MongoClient
import streamlit as st

client = MongoClient(st.secrets['url_con'])
db = client.football_data
