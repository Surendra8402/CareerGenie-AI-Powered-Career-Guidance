# Student Career Counselor Chatbot

This project is a Streamlit-based application that utilizes Google Gemini AI and Machine Learning to simulate a Student Career Counselor chatbot. Users can interact with the chatbot for career guidance, personalized skill-based recommendations, and intelligent responses.

## Setup

1. *Install Dependencies:*
   ```bash
   pip install streamlit
   pip install google.generativeai 
   pip install pandas
   pip install scikit-learn 
   ```

2. *Configure API Key:*
   Obtain an API key from the Google Generative AI service and set it in the code:
   python
   palm.configure(api_key='YOUR_API_KEY')
   

3. *Run the Application:*
   ```bash
   streamlit run app.py
   ```

  

## Usage

1. Open the application in a web browser.

2. Fill in your profile details (skills, education, experience, etc.).

3. Interact with the AI chatbot by entering career-related queries.

4. The chatbot will provide career guidance, skill-based recommendations, and empathetic responses.

## Project Structure

- AI-Councellor.py: The main script containing the Streamlit application code.
- requirements.txt: A file listing the project dependencies.

📁 AI Counselor
│── app.py                  # Main Streamlit app
│── requirements.txt         # Dependencies list
│── .streamlit/
│    └── secrets.toml        # API key storage (local use)
└── README.md                # Documentation


## Notes

- This project uses the Streamlit library for creating the web interface.
- The generative AI model from the Google Generative AI service is employed for chatbot responses.

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Generative AI Documentation](https://generativeai.dev/docs/)


