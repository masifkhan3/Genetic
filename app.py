import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

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
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel('genetic-Final.xlsx')
    return df

def preprocess_data(df):
    df = df.fillna('Unknown')
    encoders = {}
    categorical_columns = ['Disease Name', 'Gene(s) Involved', 'Inheritance Pattern', 
                         'Symptoms', 'Severity Level', 'Risk Assessment']
    
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        df[column + '_encoded'] = encoders[column].fit_transform(df[column])
    
    return df, encoders

# Load the data
df = load_data()
processed_df, encoders = preprocess_data(df)

# Train model
@st.cache_resource
def train_model():
    features = ['Disease Name_encoded', 'Gene(s) Involved_encoded', 'Inheritance Pattern_encoded',
               'Symptoms_encoded', 'Severity Level_encoded']
    X = processed_df[features]
    y = processed_df['Risk Assessment_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

rf_model = train_model()

def main():
    st.title("🧬 Genetic Disease Information System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", 
                           ["Home", "Disease Search", "Symptom Checker", "Statistics"])
    
    if page == "Home":
        show_home()
    elif page == "Disease Search":
        show_disease_search()
    elif page == "Symptom Checker":
        show_symptom_checker()
    elif page == "Statistics":
        show_statistics()

def show_home():
    st.header("Welcome to the Genetic Disease Information System")
    st.write("""
    This system helps you:
    - Search for specific genetic diseases
    - Check symptoms and find potential matching diseases
    - View statistics about genetic diseases
    """)
    
    # Display some key statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Diseases", len(df))
    with col2:
        st.metric("High Risk Diseases", 
                 len(df[df['Risk Assessment'].str.contains('High', na=False)]))
    with col3:
        st.metric("Severe Cases", 
                 len(df[df['Severity Level'].str.contains('Severe', na=False)]))

def show_disease_search():
    st.header("Disease Search")
    
    # Disease selection
    disease = st.selectbox("Select a Disease", df['Disease Name'].unique())
    
    if disease:
        disease_info = df[df['Disease Name'] == disease].iloc[0]
        
        # Display disease information in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            st.markdown(f"""
            **Disease:** {disease_info['Disease Name']}
            
            **Genes Involved:** {disease_info['Gene(s) Involved']}
            
            **Inheritance Pattern:** {disease_info['Inheritance Pattern']}
            
            **Severity Level:** {disease_info['Severity Level']}
            """)
        
        with col2:
            st.subheader("Clinical Details")
            st.markdown(f"""
            **Symptoms:** {disease_info['Symptoms']}
            
            **Risk Assessment:** {disease_info['Risk Assessment']}
            
            **Treatment Options:** {disease_info['Treatment Options']}
            """)
        
        # Additional information
        with st.expander("Medical Tests and Emergency Treatment"):
            st.write(f"**Suggested Medical Tests:** {disease_info['Suggested Medical Tests']}")
            st.write(f"**Emergency Treatment:** {disease_info['Emergency Treatment']}")

def show_symptom_checker():
    st.header("Symptom Checker")
    
    # Get user input for symptoms
    symptoms_input = st.text_area(
        "Enter symptoms (separate multiple symptoms with commas):",
        height=100
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
                        'risk': row['Risk Assessment'],
                        'treatment': row['Treatment Options']
                    })
            
            if matching_diseases:
                matching_diseases.sort(key=lambda x: x['matches'], reverse=True)
                
                st.subheader("Potential Matching Diseases")
                for i, match in enumerate(matching_diseases, 1):
                    with st.expander(f"{i}. {match['disease']} (Matches: {match['matches']})"):
                        st.markdown(f"""
                        **Disease Symptoms:** {match['symptoms']}
                        
                        **Severity Level:** {match['severity']}
                        
                        **Risk Assessment:** {match['risk']}
                        
                        **Treatment Options:** {match['treatment']}
                        """)
            else:
                st.warning("No matching diseases found for the given symptoms.")
        else:
            st.warning("Please enter symptoms to check.")

def show_statistics():
    st.header("Disease Statistics")
    
    # Create visualizations using plotly
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Severity Level Distribution")
        severity_counts = df['Severity Level'].value_counts()
        st.bar_chart(severity_counts)
    
    with col2:
        st.subheader("Risk Assessment Distribution")
        risk_counts = df['Risk Assessment'].value_counts()
        st.bar_chart(risk_counts)
    
    # Inheritance pattern distribution
    st.subheader("Inheritance Pattern Distribution")
    inheritance_counts = df['Inheritance Pattern'].value_counts()
    st.bar_chart(inheritance_counts)
    
    # Show data table
    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

if __name__ == "__main__":
    main()
