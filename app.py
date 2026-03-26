import streamlit as st

# Set the title of the app
st.title('Narrative Dominance')

# Add a sidebar for user input
st.sidebar.header('User Input')

# Add widgets for user input
def get_user_input():
    user_input = st.sidebar.text_input('Enter your narrative:')
    return user_input

# Run the app
if __name__ == '__main__':
    narrative = get_user_input()
    if narrative:
        st.write('Your narrative is:', narrative)
    else:
        st.write('Please enter a narrative in the sidebar.')