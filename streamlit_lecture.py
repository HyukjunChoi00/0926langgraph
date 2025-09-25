import streamlit as st
import pandas as pd
import numpy as np
st.title("Title")
st.subheader("sub header")
st.write("hello world")
st.write("Hi")

st.write('''# 1 제목  
### 소제목  
''')

with st.expander("expander 달기"):
    st.write('컨테이너 내용')
    st.write('''
    # 보고서
    ## 요약
    ''')

text = st.text_input("text input")
st.write(text)

gender = st.selectbox('성별', ('남자', '여자'))
st.write(f"gender: {gender}")

selected = st.checkbox("동의하시겠습니까?")

if selected:
    st.success("동의했습니다.")

with st.sidebar:
    add_radio = st.radio("언어 선택", ("한국어", "영어"))

col1, col2, col3 = st.columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")

tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)