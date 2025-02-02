import streamlit as st
import google.generativeai as genai
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

# Configure the generative AI model and API key
api_key = st.secrets["general"]["api_key"]  # Correctly accessing the key
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Title of the application
st.title("AI-Based Career Recommender")

# Initialize session state for user profile and feedback
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": "",
        "technical_skills": [],
        "non_technical_skills": [],
        "education": "",
        "experience": "",
        "certifications": [],
        "preferences": {}
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to recommend careers based on user skills, certifications, and background
def recommend_career(user_technical_skills, user_non_technical_skills, user_certifications, user_education, user_preferences):
    # Expanded dataset of careers and required skills (including non-tech careers)
    careers_data = {
        "career": ["Data Scientist", "Software Engineer", "Product Manager", "Cloud Engineer", "AI Engineer",
                   "Doctor", "Financial Analyst", "Graphic Designer", "Civil Engineer", "Marketing Manager",
                   "HR Manager", "Sales Executive", "Teacher", "Journalist", "Event Planner"],
        "required_technical_skills": [["Python", "Machine Learning", "Statistics"],
                                      ["Python", "Java", "Software Development"],
                                      ["Product Management", "Leadership", "Communication"],
                                      ["Cloud Computing", "AWS", "DevOps"],
                                      ["AI", "Machine Learning", "Deep Learning"],
                                      ["Medicine", "Patient Care", "Diagnosis"],
                                      ["Financial Modeling", "Excel", "Data Analysis"],
                                      ["Graphic Design", "Adobe Creative Suite", "Typography"],
                                      ["Structural Engineering", "AutoCAD", "Project Management"],
                                      ["Digital Marketing", "SEO", "Content Strategy"],
                                      ["HR Management", "Recruitment", "Employee Relations"],
                                      ["Sales Techniques", "CRM Software", "Negotiation"],
                                      ["Teaching Methods", "Curriculum Development", "Classroom Management"],
                                      ["Journalism", "Writing", "Editing"],
                                      ["Event Planning", "Vendor Management", "Budgeting"]],
        "required_non_technical_skills": [["Communication", "Problem Solving", "Teamwork"],
                                          ["Communication", "Time Management", "Adaptability"],
                                          ["Leadership", "Communication", "Strategic Thinking"],
                                          ["Problem Solving", "Teamwork", "Attention to Detail"],
                                          ["Critical Thinking", "Creativity", "Collaboration"],
                                          ["Empathy", "Communication", "Decision Making"],
                                          ["Analytical Thinking", "Communication", "Presentation Skills"],
                                          ["Creativity", "Communication", "Attention to Detail"],
                                          ["Problem Solving", "Teamwork", "Leadership"],
                                          ["Communication", "Creativity", "Strategic Thinking"],
                                          ["Communication", "Conflict Resolution", "Leadership"],
                                          ["Communication", "Persuasion", "Relationship Building"],
                                          ["Patience", "Communication", "Adaptability"],
                                          ["Communication", "Curiosity", "Attention to Detail"],
                                          ["Organization", "Communication", "Problem Solving"]],
        "required_certifications": [["TensorFlow Certification", "Google Cloud Certification"],
                                    ["AWS Certification", "Java Certification"],
                                    ["PMP Certification", "Scrum Master Certification"],
                                    ["AWS Certification", "Azure Certification"],
                                    ["TensorFlow Certification", "AI Certification"],
                                    ["Medical License", "Board Certification"],
                                    ["CFA Certification", "Financial Modeling Certification"],
                                    ["Adobe Certified Expert", "Graphic Design Certification"],
                                    ["PE License", "AutoCAD Certification"],
                                    ["Google Ads Certification", "HubSpot Content Marketing Certification"],
                                    ["SHRM Certification", "HR Management Certification"],
                                    ["Salesforce Certification", "Sales Training Certification"],
                                    ["Teaching License", "Curriculum Development Certification"],
                                    ["Journalism Certification", "Media Ethics Certification"],
                                    ["Event Planning Certification", "Project Management Certification"]]
    }

    df = pd.DataFrame(careers_data)

    # Filter careers based on user preferences (e.g., industry)
    if user_preferences.get("industry"):
        df = df[df["career"].str.contains(user_preferences["industry"], case=False)]

    # Convert skills and certifications to feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['required_technical_skills'].apply(lambda x: ' '.join(x)) + " " +
                                 df['required_non_technical_skills'].apply(lambda x: ' '.join(x)) + " " +
                                 df['required_certifications'].apply(lambda x: ' '.join(x)))

    # Fit a nearest neighbors model
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(X)

    user_vector = vectorizer.transform([' '.join(user_technical_skills) + " " +
                                        ' '.join(user_non_technical_skills) + " " +
                                        ' '.join(user_certifications)])
    _, indices = nn.kneighbors(user_vector)
    return df.iloc[indices[0][0]]['career']

# Function to simulate emotionally intelligent chatbot interaction
def chat_with_counselor(user_input, user_profile):
    # Include user profile data in the prompt
    prompt = f"""
    You are a career counselor with emotional intelligence. Provide empathetic and personalized advice based on the user's profile.

    User Profile:
    - Name: {user_profile["name"]}
    - Technical Skills: {', '.join(user_profile["technical_skills"])}
    - Non-Technical Skills: {', '.join(user_profile["non_technical_skills"])}
    - Education: {user_profile["education"]}
    - Experience: {user_profile["experience"]}
    - Certifications: {', '.join(user_profile["certifications"])}
    - Preferences: {user_profile["preferences"]}

    User Input: {user_input}
    """
    response = model.generate_content(prompt)
    return response.text

# Sidebar for user profile input
with st.sidebar:
    st.header("User Profile")
    st.session_state.user_profile["name"] = st.text_input("Name")
    st.session_state.user_profile["technical_skills"] = st.text_input("Technical Skills (comma-separated)").split(",")
    st.session_state.user_profile["non_technical_skills"] = st.text_input("Non-Technical Skills (comma-separated)").split(",")
    st.session_state.user_profile["education"] = st.text_input("Education")
    st.session_state.user_profile["experience"] = st.text_input("Work Experience")
    st.session_state.user_profile["certifications"] = st.text_input("Certifications (comma-separated)").split(",")
    st.session_state.user_profile["preferences"] = {
        "remote_work": st.checkbox("Prefer Remote Work"),
        "industry": st.text_input("Preferred Industry (e.g., Tech, Healthcare, Finance)")
    }

# Main chatbot interface
st.header("Career Counselor Chatbot")

# Initial chatbot message
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hello, I'm your Career Counselor. How can I help you today?"})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_message = st.chat_input("Enter your message:")

if user_message:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Generate counselor response
    if "recommend career" in user_message.lower():
        # Recommend career based on user skills, certifications, and preferences
        recommended_career = recommend_career(
            st.session_state.user_profile["technical_skills"],
            st.session_state.user_profile["non_technical_skills"],
            st.session_state.user_profile["certifications"],
            st.session_state.user_profile["education"],
            st.session_state.user_profile["preferences"]
        )
        response = f"Based on your skills, certifications, and preferences, I recommend exploring a career as a {recommended_career}."
    else:
        # Emotionally intelligent response using user profile data
        response = chat_with_counselor(user_message, st.session_state.user_profile)

    # Add counselor response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display counselor response
    with st.chat_message("assistant"):
        st.write(response)

# Feedback collection
st.header("Feedback")
feedback = st.text_area("Please provide your feedback to help us improve:")
if st.button("Submit Feedback"):
    if feedback:
        st.success("Thank you for your feedback!")