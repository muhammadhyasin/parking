# Import modules
from parkwhere import extract_all_features
import streamlit as st
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import plotly.graph_objects as go

def main():

    # Main interface
    st.title("Dashboard")
    st.write("Check out the side bar to find out more about ParkWhere!")
    
    # Sidebar interface
    st.sidebar.title("ParkWhere")
    st.sidebar.image("Homepage.jpeg", use_column_width=True)
   
    st.sidebar.header("What is ParkWhere?")
    st.sidebar.info("ParkWhere is a web application that helps you to know the parking availability")

    st.sidebar.header("How does it work?")
    how_it_works = """
    We divided available parking spaces into blocks of four zones, 'Zone 1' being the nearest.
    
    Using the collected data, We trained a binary classification model using LSTM that predicts, given a specific time, how likely it is find a parking lot.
    """
    
    
    st.sidebar.info(how_it_works)


    p1, p2 = st.columns([1, 1])

    # Get date and time inputs (defaults to current date and time)
    date_predict = p1.date_input('Enter date...', dt.date.today(), key='1') 
    time_now_predict = dt.datetime.now() + dt.timedelta(hours=8) # Based on server time
    # time_now_predict = dt.datetime.now() # Based on local machine time
    time_now_predict = time_now_predict.strftime("%H:%M")
    time_now_predict = p2.text_input("... and time", value=time_now_predict, key='1') 

    # Load trained model
    model = pickle.load(open('model.sav', 'rb'))

    # Arrange date and time inputs as a DataFrame and extract features
    predict_df = pd.DataFrame({'date_time': str(date_predict) + ' ' + time_now_predict}, index=[0])
    predict_df['date_time'] = pd.to_datetime(predict_df['date_time'])
    predict_df = extract_all_features(predict_df)
    predict_df = predict_df[['hour_min', 'day_of_week', 'ph_eve']]

    # Generate prediction and predicted probabilities
    pred = model.predict(predict_df)[0]
    pred_proba = model.predict_proba(predict_df)[0]
    
    # Create a static donut chart 
    labels = [i+"\n"+str(round(j,3)) for i, j in zip(model.classes_, pred_proba)]
    plt.figure(figsize=(6,6))
    fig = plt.pie(pred_proba, labels=labels, colors=['green', 'red'])
    my_circle = plt.Circle((0,0), 0.7, color='white')
    fig = plt.gcf()
    fig.gca().add_artist(my_circle)
    
    # Result message
    if pred == 'Zone 1':
        result_string = "Good news \U0001F973 You are likely to park at Zone 1"
        proba = round(pred_proba[1], 3)
    else:
        result_string = '\U0001F974 Oops. Looks like all nearest parking lots might have been taken up.'
        proba = round(pred_proba[0], 3)
    st.markdown(result_string)

    # Expander to view prediction results
    with st.expander('View prediction results'):
        s1, s2 = st.columns([1, 3])
        s1.markdown(f"<h3 style='text-align: center; color: white;'>{pred} (with {proba} probability)</h1>", unsafe_allow_html=True)
        s2.pyplot(fig)

if __name__ == "__main__":
    main()