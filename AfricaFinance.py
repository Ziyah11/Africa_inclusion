import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import streamlit as st
import joblib
data = pd.read_csv('Financial_inclusion_dataset.csv')
df = data.copy()

encoder = LabelEncoder()
scaler = StandardScaler()

df.drop(['country', 'uniqueid'], axis = 1, inplace = True)

for column in df.drop('bank_account', axis=1).select_dtypes(include=['object']):
    df[column] = encoder.fit_transform(df[column])


#for i in df.drop('bank_account', axis = 1).columns:
    ##if df[i].dtypes == 'O':
        #df[i] = encoder.fit_transform(df[i])

    
st.markdown("<h1 style='text-align: center; color: #151965;'>FINANCIAL INCLUSION PREDICTOR MODEL</h1>", unsafe_allow_html=True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive'>BUILT BY Ziyah</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)
st.image('pngwing.com (2).png', width = 350, use_column_width = True)
st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<p>African finance prediction plays a critical role in driving economic growth and development across the continent. By leveraging advanced data analysis techniques and predictive modeling, we can anticipate market trends, assess risk factors, and identify opportunities for investment and financial inclusion. With a focus on harnessing the vast potential of Africa's emerging markets, our predictive models aim to empower businesses, governments, and individuals to make informed financial decisions that drive sustainable development and prosperity for all.</p>", 
             unsafe_allow_html = True)
st.markdown('<br>', unsafe_allow_html = True)
st.dataframe(data, use_container_width = True)

st.sidebar.image('pngwing.com (4).png', caption = 'welcome user')

respondent_age = st.sidebar.number_input('Respondent_Age', data['age_of_respondent'].min(), data['age_of_respondent'].max())
house_hold_size = st.sidebar.number_input('houseHold_size', data['household_size'].min(), data['household_size'].max())
job_type = st.sidebar.selectbox('job_type', data['job_type'].unique())
education_level = st.sidebar.selectbox('Education_level', data['education_level'].unique())
year = st.sidebar.selectbox('YEAR', data['year'].unique())
marital_status = st.sidebar.selectbox('Marital_status', data['marital_status'].unique())
head_relationship = st.sidebar.selectbox('Head_Relationship', data['relationship_with_head'].unique())

try:
    new_education_level = encoder.transform([education_level])[0]
    new_marital_status = encoder.transform([marital_status])[0]
    new_head_relationship = encoder.transform([head_relationship])[0]
except ValueError:
    # Handle unseen label by setting a default value
    new_education_level = 0
    new_marital_status = 0
    new_head_relationship = 0

new_job_type = encoder.transform([job_type])[0]
#new_education_level = encoder.transform([education_level])[0]
#new_marital_status = encoder.transform([marital_status])[0]
#new_head_relationship = encoder.transform([head_relationship])[0]

input_var = pd.DataFrame({'age_of_respondent': [respondent_age], 'household_size':[house_hold_size], 'job_type':[new_job_type], 
                          'education_level':[new_education_level], 'year':[year], 'marital_status':[new_marital_status], 
                          'relationship_with_head' :[new_head_relationship]})
st.dataframe(input_var)
model = joblib.load('africanFinance.pkl')
prediction = st.button('Press to predict')

if prediction:
    predicted = model.predict(input_var)
    output = None
    if predicted == 1:
        output = 'Has bank_account'
    else:
        output = 'Has no Bank_account '
    st.success(f'Your prediction for the african individual {output}')
    st.balloons()




