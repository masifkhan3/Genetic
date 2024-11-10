import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Genetic Disease Information System",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache the data loading
@st.cache_data
def load_data():
    try:
        return pd.read_excel('genetic-Final.xlsx')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    df = df.fillna('Unknown')
    encoders = {}
    categorical_columns = ['Disease Name', 'Gene(s) Involved', 'Inheritance Pattern', 
                         'Symptoms', 'Severity Level', 'Risk Assessment']
    
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        df[column + '_encoded'] = encoders[column].fit_transform(df[column])
    
    return df, encoders

# Main app
def main():
    st.title("🧬 Genetic Disease Information System")

    # Load data
    df = load_data()
    
    if df is not None:
        # Preprocess data
        processed_df, encoders = preprocess_data(df)

        # Train model
        features = ['Disease Name_encoded', 'Gene(s) Involved_encoded', 'Inheritance Pattern_encoded',
                   'Symptoms_encoded', 'Severity Level_encoded']
        X = processed_df[features]
        y = processed_df['Risk Assessment_encoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Choose a page:", ["Search by Disease", "Check Symptoms"])

        if page == "Search by Disease":
            st.header("Disease Search")
            
            # Disease selection
            selected_disease = st.selectbox(
                "Select a disease:",
                sorted(df['Disease Name'].unique())
            )

            if st.button("Get Disease Information"):
                disease_info = df[df['Disease Name'] == selected_disease].iloc[0]
                
                # Display information in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Disease Details")
                    st.write(f"**Disease Name:** {disease_info['Disease Name']}")
                    st.write(f"**Genes Involved:** {disease_info['Gene(s) Involved']}")
                    st.write(f"**Inheritance Pattern:** {disease_info['Inheritance Pattern']}")
                
                with col2:
                    st.subheader("Clinical Information")
                    st.write(f"**Severity Level:** {disease_info['Severity Level']}")
                    st.write(f"**Risk Assessment:** {disease_info['Risk Assessment']}")
                
                st.subheader("Symptoms")
                st.write(disease_info['Symptoms'])
                
                st.subheader("Treatment Options")
                st.write(disease_info['Treatment Options'])

        elif page == "Check Symptoms":
            st.header("Symptom Checker")
            
            symptoms_input = st.text_area(
                "Enter symptoms (separate with commas):",
                height=100,
                placeholder="e.g., fever, headache, fatigue"
            )

            if st.button("Check Symptoms"):
                if symptoms_input:
                    symptoms_list = [s.strip().lower() for s in symptoms_input.split(',')]
                    
                    matching_diseases = []
                    for _, row in df.iterrows():
                        disease_symptoms = str(row['Symptoms']).lower()
                        matches = sum(1 for symptom in symptoms_list if symptom in disease_symptoms)
                        if matches > 0:
                            matching_diseases.append({
                                'disease': row['Disease Name'],
                                'matches': matches,
                                'symptoms': row['Symptoms'],
                                'severity': row['Severity Level'],
                                'risk': row['Risk Assessment']
                            })

                    if matching_diseases:
                        matching_diseases.sort(key=lambda x: x['matches'], reverse=True)
                        
                        st.subheader("Matching Diseases")
                        for i, match in enumerate(matching_diseases, 1):
                            with st.expander(f"{match['disease']} (Matching Symptoms: {match['matches']})"):
                                st.write(f"**Disease Symptoms:** {match['symptoms']}")
                                st.write(f"**Severity Level:** {match['severity']}")
                                st.write(f"**Risk Assessment:** {match['risk']}")
                    else:
                        st.warning("No matching diseases found for the given symptoms.")
                else:
                    st.warning("Please enter symptoms to check.")

if __name__ == "__main__":
    main()
