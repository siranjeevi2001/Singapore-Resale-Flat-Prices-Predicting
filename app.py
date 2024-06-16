import streamlit as st
import pandas as pd
import pickle
import joblib
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



# Load the saved model
# with open('trained_model2.pkl', 'rb') as f:
#     model = pickle.load(f)
    
# option extract techniq


def exgboost():

    # Title of the page
    st.header("Singapore-Resale-Flat-Prices-Predicting", divider='rainbow')
    st.write("Mean Squared Error: 757623189.5030497")
    st.write("R² Score: 0.9740141560806759")


    # model2 = joblib.load('trained_model1.pkl')

    # Collecting user inputs
    year = st.number_input('year', min_value=1990, max_value=2050, step=1, value=2000)


    type_option = {'1 ROOM': 0,
                    '2 ROOM': 1,
                    '3 ROOM': 2,
                    '4 ROOM': 3,
                    '5 ROOM': 4,
                    'EXECUTIVE': 5,
                    'MULTI GENERATION': 6,
                    'MULTI-GENERATION': 7 }
    
    selection_type = st.selectbox('select flat_type', options=list(type_option.keys()), index=0)
    flat_type = type_option[selection_type]

 
    block_option = {'1': 0,
        '10': 1,'100': 2,'101': 3,'101A': 4,'101B': 5,
        '101C': 6,'101D': 7,'102': 8,'102A': 9,'102B': 10,'102C': 11,'102D': 12,'103': 13,
        '103A': 14,'103B': 15,'103C': 16,'104': 17,'104A': 18,'104B': 19,'104C': 20,'104D': 21,
        '105': 22,'105A': 23,'105B': 24,'105C': 25,'105D': 26,'106': 27,'106A': 28,'106B': 29,
        '106C': 30,'106D': 31,'107': 32,'107A': 33,'107B': 34,'107C': 35,'107D': 36,
        '108': 37,'108A': 38,'108B': 39,'108C': 40,'109': 41 }
    
    
    selection_block = st.selectbox('Select block', options=list(block_option.keys()), index=0)
    block = block_option[selection_block]


    floor_area_sqm = st.number_input('floor_area_sqm', min_value=28,  value=28)


    street_option = {'YISHUN RING RD': 564, 'BEDOK RESERVOIR RD': 45, 'ANG MO KIO AVE 10': 14, 'ANG MO KIO AVE 3': 16, 'HOUGANG AVE 8': 213, 'TAMPINES ST 21': 460, 'BEDOK NTH ST 3': 42, 'BEDOK NTH RD': 39, 'ANG MO KIO AVE 4': 17, 'MARSILING DR': 329, 'JURONG WEST ST 42': 269, 'ANG MO KIO AVE 5': 18, 'LOR 1 TOA PAYOH': 308, 'SIMEI ST 1': 429, 'BT BATOK WEST AVE 6': 84, 'JURONG EAST ST 21': 257, 'CIRCUIT RD': 139, 'CLEMENTI AVE 4': 144, 'TAMPINES ST 22': 461, 'ANG MO KIO AVE 1': 13, 'YISHUN ST 11': 565, 'YISHUN ST 21': 567, 'JURONG WEST ST 81': 281, 'WOODLANDS ST 13': 545, 'YISHUN ST 61': 573, 'BISHAN ST 13': 55, 'UBI AVE 1': 502, 'JURONG WEST ST 52': 271, 'YISHUN ST 81': 576, 'TECK WHYE LANE': 486, 'TAMPINES ST 41': 467, 'PASIR RIS ST 21': 356, 'YISHUN AVE 6': 559, 'JURONG WEST ST 41': 268, 'BISHAN ST 12': 54, 'PASIR RIS DR 6': 352, 'SERANGOON NTH AVE 1': 420, 'PASIR RIS ST 11': 353}
    
    selection_street = st.selectbox('Select street_name', options=list(street_option.keys()), index=0)
    street_name = street_option[selection_street]
    
    
    
    town_option = {'ANG MO KIO': 0,
    'BEDOK': 1, 'BISHAN': 2,
    'BUKIT BATOK': 3,
    'BUKIT MERAH': 4, 'BUKIT PANJANG': 5,
    'BUKIT TIMAH': 6, 'CENTRAL AREA': 7,
    'CHOA CHU KANG': 8, 'CLEMENTI': 9,
    'GEYLANG': 10, 'HOUGANG': 11,
    'JURONG EAST': 12, 'JURONG WEST': 13,
    'KALLANG/WHAMPOA': 14, 'LIM CHU KANG': 15,
    'MARINE PARADE': 16, 'PASIR RIS': 17,
    'PUNGGOL': 18, 'QUEENSTOWN': 19,
    'SEMBAWANG': 20, 'SENGKANG': 21,
    'SERANGOON': 22, 'TAMPINES': 23,
    'TOA PAYOH': 24, 'WOODLANDS': 25,
    'YISHUN': 26}
    
    
    
    selection_town = st.selectbox('Select town', options=list(town_option.keys()), index=0)
    town = town_option[selection_town]
    # Creating a dictionary from the inputs
    
    input_data = {
        'year': [year],    
        'flat_type': [flat_type], 
        'block': [block],  
        'floor_area_sqm': [floor_area_sqm],  
        'town': [town],  
        'street_name': [street_name]   
    }

        
    df = pd.DataFrame(input_data)

        # Apply the same preprocessing steps to the sample data
        # (e.g., encoding, scaling, etc.)
    if st.button('Submit'):
        model = joblib.load('model\singpore_model_rf.pkl')
        # Predict the selling price using the sample data
        prediction = model.predict(df)
        st.write("Given Data:")
        st.write(df)
        st.header(f"The ReSale Price predicted is: {prediction[0]}")
            
  


def image():

    # URL you want to embed in the iframe
    tableau_url = "https://public.tableau.com/app/profile/siranjeevi.v/viz/Singapore-Resale-Flat-Prices-Predicting/Dashboard1?publish=yes"

    
    # Create a button
    if st.button('Click View Live Dashboard'):
        st.write(f'<a href="{tableau_url}" target="_blank">Click here to open Tableau Visualization</a>', unsafe_allow_html=True)

    st.image('corr.png',width=800)
    st.image('sqm.png',width=800)    
    st.image('resale_price.png',width=800)
    st.image('lease_comm.png',width=800)
    
    
    
def home():
    st.image('flat.jpg',width=800)
    
    st.write('The objective of this project is to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. This predictive model will be based on historical data of resale flat transactions, and it aims to assist both potential buyers and sellers in estimating the resale value of a flat.')
    
# Page navigation
st.sidebar.header("welcome to Singapore-Resale-Flat-Prices-Predicting", divider='rainbow')
page = st.sidebar.radio("Choose a page",
                            ["Home","Data Analysis using Tableau","weekly sale predication"],index=2)

if page == "weekly sale predication":
    exgboost()
elif page == "Data Analysis using Tableau":
    image()
elif page == "Home":
   home()
