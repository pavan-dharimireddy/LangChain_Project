
import streamlit as st
import langchain_helper

st.title("Restaurant Name Generator")
cuisine = st.sidebar.selectbox("Pick a Cuisine",["Italian","Chinese","Indian"])

'''
def generate_restaurant_name_and_items(cuisine):
    return {
        "restaurant_name": "Paneer Palace",
        "menu": ["Paneer Tikka", "Paneer Butter Masala", "Shahi Paneer", "Paneer Do Pyaza", "Kadhai Paneer", "Paneer Pasanda", "Paneer Bhurji", "Malai Kofta", "Paneer Pakora", "Paneer Tikka Masala", "Paneer Angara", "Paneer Jalfrezi", "Paneer Makhani", "Paneer Handi", "Paneer Chilli", "Paneer 65", "Paneer Biryani", "Paneer Fried Rice", "Paneer Naan", "Paneer Kulcha"]
    }


if cuisine:
    response_data = langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response_data["restaurant_name"])
    menu_items = response_data["menu"]
    st.subheader("Menu Items")
    for item in menu_items:
        st.write(f"- {item}")
'''


if cuisine:
    response_data = langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response_data["restaurant_name"].strip())
    menu_items = response_data["menu_items"].strip().split(",")
    st.subheader("Menu Items")
    for item in menu_items:
        st.write(f"- {item}")

