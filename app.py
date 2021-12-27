import streamlit as st
number = st.slider("Pick a number",0,100)

# radio button
# first argument is the title of the radio button
# second argument is the options for the ratio button
status = st.radio("Select Gender: ", ('Male', 'Female'))

# conditional statement to print
# Male if male is selected else print female
# show the result using the success function
if (status == 'Male'):
    st.success("Male")
else:
    st.success("Female")