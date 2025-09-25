### st.title('제목')  

이 코드는 웹 페이지의 제목을 설정합니다.  

st.title() 함수는 가장 큰 글씨로 제목을 표시하며, 페이지 상단에 나타납니다.

### st.session_state  

**st.session_state**는 Streamlit 애플리케이션의 세션 상태를 관리하는 특별한 객체입니다. 
Streamlit은 사용자가 웹 페이지를 새로고침하거나 위젯을 조작할 때마다 스크립트 전체를 위에서부터 아래로 다시 실행합니다. 이 때문에 변수에 저장된 값들은 리프레시될 때마다 초기화됩니다.

- st.session_state는 이러한 초기화를 방지하고, 사용자 세션 동안 값을 유지할 수 있도록 해줍니다.

- st.session_state는 딕셔너리와 유사하게 작동합니다.

- if "messages" not in st.session_state:: "messages"라는 키가 세션 상태에 존재하지 않으면(즉, 앱이 처음 실행될 때),

st.session_state.messages = []: 빈 리스트를 생성하여 st.session_state에 messages라는 키로 저장합니다. 

이 리스트는 사용자와 AI의 대화 기록을 담는 역할을 합니다.

채팅 기록 표시 및 사용자 입력 처리
for message in st.session_state.messages:: messages 리스트에 저장된 이전 대화 기록을 반복해서 가져와 화면에 표시합니다.

with st.chat_message(message["role"]):: st.chat_message는 Streamlit에서 제공하는 챗봇 전용 컨테이너입니다. role에 따라 사용자(user) 또는 어시스턴트(assistant) 아이콘과 함께 말풍선 스타일로 메시지를 표시해 줍니다.

st.markdown(message["content"]): 각 메시지의 내용을 Markdown 형식으로 변환하여 화면에 렌더링합니다.

사용자 입력 및 LLM 응답 처리
if prompt := st.chat_input("무엇을 도와드릴까요?"):: 바다코끼리 연산자(:=)를 사용하여 st.chat_input 위젯에서 사용자가 입력한 내용을 prompt 변수에 할당하는 동시에, 입력 내용이 있는지(빈 문자열이 아닌지) 확인합니다.

st.session_state.messages.append(...): 사용자가 메시지를 입력하면, 해당 메시지를 st.session_state.messages 리스트에 추가하여 대화 기록을 업데이트합니다.

with st.spinner("생각 중..."):: LLM이 응답을 생성하는 동안 "생각 중..."이라는 로딩 메시지를 표시하여 사용자에게 기다려달라는 신호를 줍니다.

response = llm_chain.invoke(...): LangChain의 LLMChain을 사용하여 사용자의 질문(prompt)을 LLM에게 전달하고 응답을 받습니다.

full_response = response["text"]: LangChain의 응답 객체에서 실제 텍스트 내용을 추출합니다.

st.session_state.messages.append(...): AI의 응답이 생성되면, 이 역시 st.session_state.messages 리스트에 추가하여 대화 기록을 보존합니다.

결론적으로, 이 코드는 st.session_state를 활용하여 챗봇의 대화 기록을 사용자 세션 내내 유지하고, st.chat_message와 같은 챗봇 전용 위젯을 이용해 대화형 UI를 구현하는 전형적인 Streamlit 챗봇 구조를 보여줍니다.
