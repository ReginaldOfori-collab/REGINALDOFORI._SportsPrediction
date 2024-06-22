import streamlit as st
import pickle
import pandas as pd


try:
    with open('best_model2.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.title("Football Predictions")

def features():
    attribute1 = st.number_input("Player's Movement Reaction")
    attribute2 = st.number_input("PLayer's Potential")
    attribute3 = st.number_input("Player's Wage in Euros")
    attribute4 = st.number_input("Player's Value in Euros")
    attribute5 = st.number_input("Player's Passing")
    attribute6 = st.number_input("Player's Attacking Short Passing")
    attribute7 = st.number_input("Player's Mentality Vision")
    attribute8 = st.number_input("Player's International Reputation")
    attribute9 = st.number_input("Player's Skill Long Passing")
    attribute10 = st.number_input("Player's Power Shot")
    attribute11 = st.number_input("Player's Physic")
    attribute12 = st.number_input("Player's Age")
    attribute13 = st.number_input("Player's Skill Ball Control")
    attribute14 = st.number_input("Player's Dribbling")
    attribute15 = st.number_input("Player's Skill Curve")
    attribute16 = st.number_input("Player's Power Long Shots")
    attribute17 = st.number_input("Player's Shooting")
    attribute18 = st.number_input("Player's Mentality Aggression")
    attribute19 = st.number_input("Player's Attacking Crossing")
    attribute20 = st.number_input("Player's Skill Freekick Accuracy")

    dict = {
        'movement_reactions': attribute1,
        'potential': attribute2,
        'wage_eur': attribute3,
        'value_eur': attribute4,
        'passing': attribute5,
        'attacking_short_passing': attribute6,
        'mentality_vision': attribute7,
        'international_reputation': attribute8,
        'skill_long_passing': attribute9,
        'power_shot_power': attribute10,
        'physic': attribute11,
        'age': attribute12,
        'skill_ball_control': attribute13,
        'dribbling': attribute14,
        'skill_curve': attribute15,
        'power_long_shots': attribute16,
        'shooting': attribute17,
        'mentality_aggression': attribute18,
        'attacking_crossing': attribute19,
        'skill_fk_accuracy': attribute20
    }

    df = pd.DataFrame(dict, index = [0])
    return df

input_data = features()

try:
    scaledinput = scaler.transform(input_data)

    prediction = model.predict(scaledinput)

    st.subheader('Prediction')
    st.write(f'Predicted Rating: {prediction[0]}')

except Exception as e:
    st.error(f'Error in prediction: {e}')
