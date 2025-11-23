import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Pet Adoption Predictor",
    page_icon="🐾",
    layout="wide"
)

# Title and description
st.title("🐾 Pet Adoption Predictor")
st.markdown("""
This app predicts whether a pet is likely to be adopted based on its characteristics. 
If the prediction indicates low adoption likelihood, we'll provide personalized suggestions to improve adoption chances.
""")

def build_preprocessor():
    """
    Build the preprocessing pipeline (same as in your training code)
    """
    numeric_features = ['AgeMonths', 'WeightKg', 'TimeInShelterDays', 'AdoptionFee']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'),
             ['PetType', 'Breed', 'Color']),
            ('ordinal', OrdinalEncoder(categories=[['Small', 'Medium', 'Large']]),
             ['Size']),
            ('scaler', StandardScaler(), numeric_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

@st.cache_resource
def create_and_train_model():
    """
    Create and train the model using the actual dataset
    """
    try:
        # Load the actual dataset
        try:
            df = pd.read_csv("./data/pet_adoption_data.csv")
            st.success("✅ Real dataset loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Dataset file not found. Please ensure './data/pet_adoption_data.csv' exists.")
            return None
        
        # Prepare the data (same as your training code)
        df_clean = df.drop(columns=["PetID"], errors="ignore")
        X = df_clean.drop(columns=["AdoptionLikelihood"])
        y = df_clean["AdoptionLikelihood"]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create the pipeline with the exact same parameters as your final model
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
        # Train the model on the real data
        with st.spinner("Training model on real adoption data..."):
            pipeline.fit(X_train, y_train)
        
        # Calculate accuracy
        train_accuracy = pipeline.score(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)
        
        st.success(f"✅ Model trained successfully! (Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f})")
        st.info(f"📊 Dataset info: {len(df)} records, Adoption rate: {y.mean():.1%}")
        
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Error creating/training model: {e}")
        return None

def get_improvement_recommendations(pet_data):
    """
    Generate personalized improvement recommendations based on pet characteristics
    """
    recommendations = []
    
    # Vaccination recommendation
    if pet_data['Vaccinated'] == 0:
        recommendations.append("💉 **Get the pet vaccinated**: Vaccinated pets are much more likely to be adopted based on our data")
    
    # Health condition recommendation
    if pet_data['HealthCondition'] == 1:
        recommendations.append("🏥 **Address medical conditions**: Consider treatment or create a compelling story about the pet's resilience and care needs")
    
    # Age-related recommendations
    if pet_data['AgeMonths'] > 24:  # Older than 2 years
        recommendations.append("🎯 **Highlight maturity benefits**: Emphasize that older pets are often calmer, trained, and require less supervision than puppies/kittens")
    
    # Time in shelter recommendations
    if pet_data['TimeInShelterDays'] > 60:
        recommendations.append("📸 **Update photos and create video content**: Pets with longer stays need fresh, engaging profiles. Consider a 'featured pet' promotion")
    elif pet_data['TimeInShelterDays'] > 30:
        recommendations.append("🌟 **Boost visibility**: Share on social media with engaging stories about the pet's personality")
    
    # Size-specific recommendations
    if pet_data['Size'] == 'Large':
        recommendations.append("🏠 **Showcase space requirements**: Provide information about suitable living situations and highlight loyalty traits of large pets")
    elif pet_data['Size'] == 'Small':
        recommendations.append("🏢 **Highlight apartment suitability**: Emphasize that small pets are great for apartments and require less space")
    
    # Weight recommendations
    if pet_data['WeightKg'] > 20 and pet_data['PetType'] in ['Dog', 'Rabbit']:
        recommendations.append("⚖️ **Provide weight management info**: Share exercise routines and diet plans to show the pet can be healthy and active")
    
    # Adoption fee recommendations
    if pet_data['AdoptionFee'] > 370:
        recommendations.append("💰 **Consider fee adjustments or promotions**: High adoption fees can deter potential adopters. Consider 'fee-waived' events or sponsorship programs")
    elif pet_data['AdoptionFee'] < 120:
        recommendations.append("💰 **Highlight value proposition**: Emphasize that the low fee includes vaccinations, health checks, and microchipping")
    
    # Previous owner recommendations
    if pet_data['PreviousOwner'] == 1:
        recommendations.append("📝 **Share positive history**: Highlight that the pet is familiar with home environments and may be house-trained")
    
    # Breed-specific recommendations
    if pet_data['Breed'] in ['Golden Retriever', 'Labrador']:
        recommendations.append("👨‍👩‍👧‍👦 **Highlight family-friendly nature**: Emphasize their reputation as great family pets")
    elif pet_data['Breed'] in ['Persian', 'Siamese']:
        recommendations.append("😻 **Showcase personality traits**: Highlight breed-specific characteristics that appeal to cat lovers")
    
    # Critical recommendations for low adoption likelihood
    recommendations.extend([
        "🎥 **Create engaging video content**: Show the pet's personality through short, professional videos",
        "🤝 **Organize meet-and-greet events**: Let potential adopters interact with the pet in a positive setting",
        "🌟 **Feature as 'Pet of the Week'**: Give the pet extra visibility on your website and social media"
    ])
    
    return recommendations

def predict_adoption_likelihood(model, input_data):
    """
    Make prediction using the trained model - returns 1 (adopted) or 0 (not adopted)
    """
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict class (0 or 1)
        prediction = model.predict(input_df)[0]
        
        # Also get probability for more insight
        probability = model.predict_proba(input_df)[0][1]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def main():
    # Create and train model
    model = create_and_train_model()
    
    if model is None:
        st.error("Failed to create the model. Please check if the dataset file exists at './data/pet_adoption_data.csv'")
        return
    
    # Sidebar for user input
    st.sidebar.header("📋 Pet Information")
    
    # Pet Type
    pet_type = st.sidebar.selectbox(
        "Pet Type *",
        ["Dog", "Cat", "Bird", "Rabbit"],
        help="Select the type of pet"
    )
    
    # Breed options based on pet type
    breed_options = {
        "Dog": ["Golden Retriever", "Labrador", "Poodle"],
        "Cat": ["Persian", "Siamese"],
        "Bird": ["Parakeet"],
        "Rabbit": ["Rabbit"]
    }
    
    breed = st.sidebar.selectbox(
        "Breed *",
        breed_options[pet_type],
        help="Select the specific breed"
    )
    
    # Other inputs in columns
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age_months = st.slider("Age (months) *", 1, 200, 24, 
                              help="Age of the pet in months")
        weight_kg = st.slider("Weight (kg) *", 0.1, 50.0, 10.0, 0.1,
                             help="Weight of the pet in kilograms")
        size = st.selectbox("Size *", ["Small", "Medium", "Large"],
                           help="Size category of the pet")
    
    with col2:
        color = st.selectbox("Color *", 
                            ["Orange", "White", "Gray", "Brown", "Black"],
                            help="Primary color of the pet")
        time_in_shelter = st.slider("Days in Shelter *", 1, 365, 30,
                                   help="Number of days the pet has spent in the shelter")
        adoption_fee = st.slider("Adoption Fee ($) *", 0, 500, 100,
                                help="Adoption fee charged in USD")
    
    # Binary inputs
    st.sidebar.subheader("Health & History")
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        vaccinated = st.radio("Vaccinated *", [1, 0], 
                             format_func=lambda x: "Yes" if x == 1 else "No",
                             help="Whether the pet has been vaccinated")
        health_condition = st.radio("Health Condition *", [0, 1], 
                                   format_func=lambda x: "Healthy" if x == 0 else "Medical Condition",
                                   help="Current health status of the pet")
    
    with col4:
        previous_owner = st.radio("Previous Owner *", [0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes",
                                 help="Whether the pet had a previous owner")
    
    # Create input data dictionary
    input_data = {
        'PetType': pet_type,
        'Breed': breed,
        'AgeMonths': age_months,
        'Color': color,
        'Size': size,
        'WeightKg': weight_kg,
        'Vaccinated': vaccinated,
        'HealthCondition': health_condition,
        'TimeInShelterDays': time_in_shelter,
        'AdoptionFee': adoption_fee,
        'PreviousOwner': previous_owner
    }
    
    # Display input summary
    st.subheader("📊 Pet Profile Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Type:** {pet_type}")
        st.write(f"**Breed:** {breed}")
        st.write(f"**Age:** {age_months} months ({age_months/12:.1f} years)")
        st.write(f"**Color:** {color}")
        st.write(f"**Size:** {size}")
    
    with col2:
        st.write(f"**Weight:** {weight_kg} kg")
        st.write(f"**Vaccinated:** {'Yes' if vaccinated == 1 else 'No'}")
        st.write(f"**Health:** {'Healthy' if health_condition == 0 else 'Medical Condition'}")
        st.write(f"**Days in Shelter:** {time_in_shelter}")
        st.write(f"**Adoption Fee:** ${adoption_fee}")
        st.write(f"**Previous Owner:** {'Yes' if previous_owner == 1 else 'No'}")
    
    # Prediction section
    st.subheader("🎯 Adoption Prediction")
    
    if st.button("Predict Adoption Likelihood", type="primary", use_container_width=True):
        with st.spinner("Analyzing adoption likelihood..."):
            # Make prediction
            prediction, probability = predict_adoption_likelihood(model, input_data)
            
            if prediction is not None:
                # Display result
                st.markdown("---")
                
                if prediction == 1:
                    st.success("🎉 **High Adoption Likelihood**")
                    st.metric(
                        label="Prediction",
                        value="LIKELY TO BE ADOPTED",
                        delta=f"Confidence: {probability:.1%}"
                    )
                    
                    st.subheader("✅ Tips to Maintain High Adoption Chances")
                    success_tips = [
                        "📸 **Keep photos updated** - Continue sharing fresh, high-quality images",
                        "📝 **Refresh the description** - Regularly update the pet's profile with new personality insights",
                        "🎥 **Share video updates** - Show the pet interacting and being social",
                        "🌟 **Leverage positive traits** - Emphasize the characteristics that make this pet special"
                    ]
                    
                    for tip in success_tips:
                        st.write(f"• {tip}")
                        
                else:  # prediction == 0
                    st.error("📉 **Low Adoption Likelihood**")
                    st.metric(
                        label="Prediction", 
                        value="UNLIKELY TO BE ADOPTED",
                        delta=f"Confidence: {1-probability:.1%}",
                        delta_color="inverse"
                    )
                    
                    st.subheader("💡 Personalized Recommendations to Improve Adoption Chances")
                    
                    recommendations = get_improvement_recommendations(input_data)
                    
                    for i, recommendation in enumerate(recommendations, 1):
                        st.write(f"{i}. {recommendation}")
                    
                    st.info("💡 **Tip:** Implementing even 2-3 of these recommendations can significantly improve adoption chances!")

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this prediction:** 
    - Based on machine learning analysis of 2,007 pet adoption records
    - Model: Random Forest Classifier (n_estimators=200, max_depth=12)
    - Model accuracy: ~92% on test data
    - Factors considered: All input characteristics above
    - *For shelter use: Always combine with staff assessment*
    """)

if __name__ == "__main__":
    main()