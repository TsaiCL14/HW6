# https://docs.streamlit.io/en/stable/api.html#streamlit.slider
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
# load model


@st.cache_data
def load_model():
    clf = joblib.load('model_hw6.joblib')
    scaler = joblib.load('scaler_hw6.joblib')
    return clf, scaler


def convert_CryoSleep(sex1):
    return 1 if CryoSleep == 'True' else 0


# dict1 = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
dict_HomePlanet = {'Earth': 0, 'Europa': 1, 'Mars': 2}
dict_Destination = {'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2}
dict_deck = {'B': 0, 'F': 1, 'A': 2, 'G': 3, 'E': 4, 'D': 5, 'C': 6, 'T': 7}
dict_side = {'P': 0, 'S': 1}


def convert_HomePlanet(HomePlanet):
    return dict_HomePlanet[HomePlanet]


def convert_Destination(Destination):
    return dict_Destination[Destination]


def convert_deck(deck):
    return dict_deck[deck]


def convert_side(side):
    return dict_side[side]


clf, scaler = load_model()

# 畫面設計
st.markdown('# 生還預測系統')
HomePlanet_series = pd.Series(['Earth', 'Europa', 'Mars'])
CryoSleep_series = pd.Series(['True', 'False'])
Destination_series = pd.Series(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'])
deck_series = pd.Series(['B', 'F', 'A', 'G', 'E', 'D', 'C', 'T'])
side_series = pd.Series(['P', 'S'])

# '家鄉星球:', HomePlanet
HomePlanet = st.sidebar.radio('家鄉星球:', HomePlanet_series)

# '假死狀態(是否):', CryoSleep
CryoSleep = st.sidebar.radio('假死狀態:', CryoSleep_series)

# '目的星球:', Destination
Destination = st.sidebar.radio('家鄉星球:', Destination_series)

# '年齡:', Age
Age = st.sidebar.slider('年齡', 0, 100, 20)

# '客房服務消費金額:', RoomService
RoomService = st.sidebar.slider('客房服務消費金額', 0, 10000000, 0)

# '美食街消費金額:', FoodCourt
FoodCourt = st.sidebar.slider('美食街消費金額', 0, 10000000, 0)

# '購物中心消費金額:', ShoppingMall
ShoppingMall = st.sidebar.slider('購物中心消費金額', 0, 10000000, 0)

# 'SPA消費金額:', Spa
Spa = st.sidebar.slider('SPA消費金額', 0, 10000000, 0)

# 'VR消費金額:', VRDeck
VRDeck = st.sidebar.slider('VR消費金額', 0, 10000000, 0)

# '床位(甲板位置):', deck
deck = st.sidebar.radio('床位(甲板位置):', deck_series)

# '床位(編號):', num
num = st.sidebar.slider('床位(編號)', 0, 1894, 0)

# '床位(左舷or右舷):', side
side = st.sidebar.radio('床位(左舷or右舷):', side_series)


# '艙等:', pclass
# pclass = st.sidebar.selectbox('艙等:', pclass_series)


# st.image('./TITANIC.png')

if st.sidebar.button('預測'):
    # predict
    X = []
    # pclass	sex	age	sibsp	parch	fare	adult_male	embark_town
    # adult_male = 1 if age >= 20 and sex == '男性' else 0
    X.append([convert_HomePlanet(HomePlanet), convert_CryoSleep(CryoSleep), convert_Destination(Destination),
              Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,
              convert_deck(deck), num, convert_side(side)])
    X = scaler.transform(np.array(X))

    if clf.predict(X) == 1:
        st.markdown(f'### ==> **生還, 生存機率={clf.predict_proba(X)[0][1]:.2%}**')
    else:
        st.markdown(f'### ==> **失蹤, 生存機率={clf.predict_proba(X)[0][1]:.2%}**')
