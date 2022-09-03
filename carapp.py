import streamlit as st
import pickle
import numpy as np

# import the model
pip = pickle.load(open('pip.pkl','rb'))
data = pickle.load(open('data.pkl','rb'))


st.title("Car Price Predictor")

# fueltype
ftype = st.selectbox('FuelType',data['fueltype'].unique())

# aspiration
asp = st.selectbox('Aspiration',data['aspiration'].unique())

# carbody
cbody = st.selectbox('CarBody',data['carbody'].unique())

# drivewheel
wheel = st.selectbox('DriveWheel',data['drivewheel'].unique())

# enginetype
etype = st.selectbox('EngineType',data['enginetype'].unique())

# stroke
stroke = st.selectbox('Stroke',[2,4])

# wheelbase
base = st.number_input('WheelBase of the Car')

# curbweight
weight = st.number_input('CurbWeight of the Car')

# lenght
length = st.number_input('Lenght of the Car')

# width
width = st.number_input('Width of the Car')

# height
height = st.number_input('Height of the Car')

# enginesize
size = st.number_input('EngineSize of the Car')

# boreratio
ratio = st.number_input('BoreRatio of the Car')

# horsepower
power = st.number_input('HorsePower of the Car')

# citympg
cmpg = st.number_input('Citympg of the Car')

# highwaympg
hmpg = st.number_input('Highwaympg of the Car')

if st.button('Predict Price'):
    mpg = None
    volume = None
    mpg=(cmpg+hmpg)/2
    volume=length*width*height
    query = np.array([ftype,asp,cbody,wheel,base,weight,etype,size,ratio,power,stroke,volume,mpg])

    query = query.reshape(1,13)
    st.title("The predicted price of this configuration is " + str(int(pip.predict(query)[0])))
