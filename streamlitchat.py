import streamlit as st

st.title('제목')

from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ["GOOGLE_API_KEY"] = ''
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{question}")
llm_chain = LLMChain(llm=llm, prompt=prompt)

# 채팅 기록(Chat history) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기

# := 의 의미는 아래의 코드를 합친 것과 같음. (Walrus Operator)
# prompt = st.chat_input("무엇을 도와드릴까요?")
# if prompt:

if prompt := st.chat_input("무엇을 도와드릴까요?"):
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LLM 응답 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            # LLM 체인 실행
            response = llm_chain.invoke({"question": prompt})
            full_response = response["text"]
            st.markdown(full_response)
    
    # AI 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})
