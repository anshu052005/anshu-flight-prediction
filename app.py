import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load model, scaler, and encoders
@st.cache_resource
def load_artifacts():
    try:
        # Try to load from current directory first
        model = joblib.load('flight_fare_model.joblib')
        scaler = joblib.load('scaler.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure all .joblib files are uploaded.")
        return None, None, None

def main():
    # Load artifacts
    model, scaler, label_encoders = load_artifacts()
    
    if model is None:
        st.stop()
    
    # Centered title with Indian theme color
    st.markdown("<h1 style='text-align: center; color: #FF9933;'>✈️ Flight Fare Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Enter flight details to predict the fare in Indian Rupees</p>", unsafe_allow_html=True)
    
    # Define the actual categorical values in correct order
    airline_options = ['Air India', 'AirAsia', 'GoAir', 'IndiGo', 'SpiceJet', 'Vistara']
    city_options = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
    class_options = ['Economy', 'Business']
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        airline = st.selectbox("✈️ Airline", airline_options)
        source_city = st.selectbox("🛫 Source City", city_options)
        destination_city = st.selectbox("🛬 Destination City", city_options)
        departure_time = st.text_input("🕐 Departure Time (HH:MM, 24hr format)", "10:00")
        arrival_time = st.text_input("🕕 Arrival Time (HH:MM, 24hr format)", "12:00")
    
    with col2:
        duration = st.number_input("⏱️ Duration (in hours)", min_value=0.1, max_value=24.0, value=2.0, step=0.1)
        total_stops = st.selectbox("🛑 Total Stops", [0, 1])
        flight_class = st.selectbox("💺 Class", class_options)
        journey_date = st.date_input("📅 Date of Journey")
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Fare", use_container_width=True)
    
    if predict_button:
        # Validation
        if source_city == destination_city:
            st.error("❌ Source and destination cities cannot be the same!")
            return
            
        try:
            # Extract date features
            journey_month = journey_date.month
            journey_day = journey_date.day
            journey_dayofweek = journey_date.weekday()
            
            # Extract time features
            dep_time = pd.to_datetime(departure_time, format='%H:%M')
            arr_time = pd.to_datetime(arrival_time, format='%H:%M')
            departure_hour = dep_time.hour
            arrival_hour = arr_time.hour
            
            # Manually encode based on the predefined order
            airline_enc = airline_options.index(airline)
            source_city_enc = city_options.index(source_city)
            destination_city_enc = city_options.index(destination_city)
            class_enc = class_options.index(flight_class)
            
            # Prepare input array
            input_data = np.array([
                airline_enc,
                source_city_enc,
                destination_city_enc,
                duration,
                total_stops,
                class_enc,
                journey_month,
                journey_day,
                journey_dayofweek,
                departure_hour,
                arrival_hour
            ]).reshape(1, -1)
            
            # Scale and predict
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)
            
            # Display result with styling
            st.success(f"🎯 **Predicted Flight Fare: ₹{prediction[0]:,.2f}**")
            
            # Additional info
            st.info(f"""
**Flight Details:**
- **Route:** {source_city} → {destination_city}
- **Airline:** {airline}
- **Class:** {flight_class}
- **Duration:** {duration} hours
- **Stops:** {total_stops}
            """)
            
        except ValueError as ve:
            st.error(f"❌ Invalid time format. Please use HH:MM format (e.g., 14:30)")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()