# main_app.py
import streamlit as st
from tp1_trees_graphs import show_tp1
from tp2_tree_operations import show_tp2
from tp3_heap_sort import show_tp3
# Page configuration
st.set_page_config(page_title="TP ALGO", layout="wide")

# Centered title (always shown)
st.markdown(
    "<h1 style='text-align: center; color: #1f77b4;'>TP ALGO</h1>",
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("Navigation")
selected_tp = st.sidebar.radio(
    "Select TP:",
    ["Accueil", "TP1", "TP2", "TP3", "TP4", "TP5", "TP6"],
)

# Main content based on selection
if selected_tp == "Accueil":
    # Home Page
    st.markdown(
        "<h2 style='text-align: center; color: #2ca02c;'>Welcome to TP ALGO Project</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; font-size: 24px; color: #333; margin: 40px 0;'>Supervising Professor:</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<li>Aroussi Sanaa</li>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 24px; color: #333; margin: 40px 0;'>Team Members:</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<ul style='text-align: center; font-size: 20px; color: #555; list-style: none; padding: 0;'>",
        unsafe_allow_html=True,
    )
    st.markdown("<li>Sidi Moussa Safia</li>", unsafe_allow_html=True)
    st.markdown("<li>Laraba Nada</li>", unsafe_allow_html=True)
    st.markdown("<li>Aida Douaa</li>", unsafe_allow_html=True)
    st.markdown("<li>Harizi Khouloud</li>", unsafe_allow_html=True)
    st.markdown("<li>Abdallah Yassmine</li>", unsafe_allow_html=True)

    st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 16px; color: #666; margin-top: 40px;'>Use the sidebar to select a TP and explore the algorithms!</p>",
        unsafe_allow_html=True,
    )

elif selected_tp == "TP1":
    show_tp1()
elif selected_tp == "TP2":
    show_tp2()
elif selected_tp == "TP3":
    show_tp3()

# Placeholder for other TPs
else:
    st.info(
        f"{selected_tp} content coming soon. "
    )