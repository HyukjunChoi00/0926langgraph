import json
import asyncio
from typing import Annotated
from typing import TypedDict, Annotated, List, Dict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from playwright.async_api import async_playwright
import os
os.environ["GOOGLE_API_KEY"] = ''

# 윈도우 사용자의 경우 아래 윈도우 호환성 정책 변경 코드
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

#############################  패키지 import

############# streamlit
import streamlit as st

st.title('이름 학번 검색 Agent')

#################


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


##### 각 문법 설명 #####

# TypedDict : 더 엄격한 타입 검사, 타입 힌트 제공
# Annotated : 타입 힌트에 메타데이터를 추가  예) Annotated[Type, metadata1, metadata2, ...]
# Annotated : LangGraph에서는 Annotated를 사용하여 특별한 동작을 정의
# add_messages : 메시지를 리스트에 추가

##################################################################

# 상태 정의

class State(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    extracted_keyword: str
    expanded_keywords: list[str]
    search_results: list[dict]
    answer: str

##################################################################

# 키워드 추출 함수 (아직 노드로 정의하진 않았고, chain을 만들어둠)

def build_keyword_extraction_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 사용자의 질문에서 핵심 검색 키워드를 추출하는 Assistant이다."),
         ("user", "다음 사용자의 질문에서 검색에 사용할 핵심 키워드만 추출하라. 반드시 영문 키워드로 제공하라.\n"
                  "예시:\n"
                  "- 'NVIDIA 주식 동향 조사좀 해줘' → 'NVIDIA'\n"
                  "- '테슬라 주가 전망이 어때?' → 'Tesla'\n"
                  "- '애플 아이폰 최신 소식 알려줘' → 'Apple'\n"
                  "- 'AI 기술 발전 현황' → 'AI'\n"
                  "질문: {query}\n\n"
                  "다음 JSON 형식으로만 응답해줘:\n"
                  '{{"extracted_keyword": "추출된_키워드"}}\n'
                  "핵심 키워드:")]
    )
    
    return prompt_template | llm

##################################################################

# 쿼리 확장 함수 (chain)

def build_query_expansion_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 키워드형 Query Expansion Assistant이다. 주어진 키워드와 관련된 산업, 기술, 시장 키워드로 확장해야 한다."),
         ("user", "다음 키워드를 확장하라:\n"
                  "원본 키워드: {query}\n"
                  "확장할 개수: {n}개\n\n"
                  "확장 규칙:\n"
                  "- Tesla → electric vehicle, EV, battery\n"
                  "- NVIDIA → AI, GPU, semiconductor\n"
                  "- Apple → iPhone, technology, consumer electronics\n"
                  "- Microsoft → cloud computing, software, Azure\n\n"
                  "'{query}' 키워드에 대해 관련 산업/기술 키워드 {n}개를 영문으로 생성하라.\n\n"
                  "반드시 다음 JSON 형식으로만 응답하라 (다른 텍스트 포함 금지):\n"
                  '{{"expanded_search_query_list": ["키워드1", "키워드2"]}}\n')]
    )
    
    return prompt_template | llm

##################################################################

# LLM을 이용한 답변 생성
# 아래 코드 중 state.get("search_results ... )를 통해, 이전 노드에 저장된 반환값인 search_results 값을 가져올 수 있음.
# 가져온 값을 context라는 변수에 저장
# context를 prompt에 삽입
# 즉, LLM이 context를 보고 답을 할 수 있음.
# 답변을 markdown 방식으로 수행하여, 이후 Streamlit에서 볼 답변이 더 구조도 있게 보이도록

def generate_response(state):
    context = state.get("search_results", "")
    
    curr_human_turn = HumanMessage(content=f"질문: {state['query']}\n"
                            f"검색 결과:\n```\n{context}```"
                             "\n---\n"
                             "응답은 markdown을 이용해 리포트 스타일로 한국어로 응답해라. "
                             "사용자의 질문의 의도에 맞는 정답 부분을 강조해라.")
    messages = state["messages"] + [curr_human_turn]
    response = llm.invoke(messages)

    return {"messages": [*messages, response],
            "answer": response.content}

##################################################################

def parse_json_response(response) -> dict:
    """JSON 응답을 파싱하는 간단한 함수"""

    # AIMessage 객체에서 content 추출
    content = str(getattr(response, 'content', response))

    # JSON 블록 찾기
    if '```json' in content:
        start = content.find('```json') + 7
        end = content.find('```', start)
        json_str = content[start:end].strip()
    elif '{' in content and '}' in content:
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]
    else:
        return {"extracted_keyword": "", "expanded_search_query_list": []}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"extracted_keyword": "", "expanded_search_query_list": []}

##################################################### 노드 생성 #################################################################

# 키워드 추출

def extract_keyword(state):
    """핵심 키워드 추출 노드"""
    print(f"🔍 키워드 추출 중: {state['query']}")
    
    keyword_extraction_chain = build_keyword_extraction_chain()
    original_query = state["query"]
    response = keyword_extraction_chain.invoke({"query": original_query})
    parsed_response = parse_json_response(response)
    
    extracted_keyword = parsed_response.get("extracted_keyword", "")
    print(f"✅ 추출된 키워드: {extracted_keyword}")
    
    return {"extracted_keyword": extracted_keyword}

##################################################################

def query_expansion(state):
    """쿼리 확장 노드"""
    print(f"🔄 쿼리 확장 중: {state['extracted_keyword']}")
    
    query_expansion_chain = build_query_expansion_chain()
    extracted_keyword = state["extracted_keyword"]
    response = query_expansion_chain.invoke({"query": extracted_keyword, "n": 2})
    parsed_response = parse_json_response(response)
    
    expanded_keywords = parsed_response.get("expanded_search_query_list", [])
    # 원본 키워드도 포함
    all_keywords = [extracted_keyword] + expanded_keywords
    print(f"✅ 확장된 키워드: {all_keywords}")
    
    return {"expanded_keywords": all_keywords}

#################################################################

# 검색 노드
# 확장된 키워드 목록을 받고
# 각 키워드에 대한 검색을 수행
# 이후 결과값을 all_results에 정리 후, search results로 반환환

async def search_news(state):
    """뉴스 검색 노드"""
    print(f"📰 뉴스 검색 중...")
    
    expanded_keywords = state.get("expanded_keywords", [])  # 키가 없으면 기본값인 빈 리스트 [] 반환. KeyError 방지.
    all_results = []
    for query in expanded_keywords:
        results = await scrape_articles_with_content(query)
        results = results['search_results']

        for i in results :
            title = i['title']
            url = i['url']
            content = i['content']
            all_results.extend([{
                    "title": title,
                    "url": url,
                    "content": content,
                    "search_query": query
                }])
    

    
    return {"search_results": all_results}



async def scrape_articles_with_content(query: str, max_articles: int = 3) -> str:
    """
    Econotimes에서 관련 기사 제목, URL, 본문을 스크랩하는 비동기 함수
    사용자 특정 종목에 대한 동향 분석 등을 요청할 때 이 도구를 이용해 뉴스 기사를 검색합니다.
    
    
    Args:
        query: 검색할 키워드 (예: tesla, apple, bitcoin 등)
        max_articles: 추출할 최대 기사 수 (기본값: 3)
    
    Returns:
        기사 정보가 포함된 JSON 형태의 문자열
    """
    print(f"🚀 Econotimes에서 '{query}' 검색 중...")
    
    output_list = []
    
    try:
        async with async_playwright() as p:
            # 브라우저 실행
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # 검색 페이지로 이동
            search_url = f"https://econotimes.com/search?v={query}&search="
            await page.goto(search_url)
            await asyncio.sleep(2)
            
            # XPath를 사용해서 모든 기사 제목 요소 찾기
            general_xpath = '//*[@id="archivePage"]/div/div[2]/div/p[1]/a'
            elements = await page.locator(f"xpath={general_xpath}").all()
            
            
            if not elements:
                await browser.close()
                return f"'{query}'에 대한 기사를 찾을 수 없습니다."
            
            # 지정된 개수만큼 기사 처리
            for i, element in enumerate(elements[:max_articles], 1):
                try:
                    # 기사 제목과 링크 추출
                    title = await element.text_content()
                    href = await element.get_attribute('href')
                    
                    if title and href:
                        title = title.strip()
                        full_url = f"https://econotimes.com{href}" if href.startswith('/') else href
                        
                        print(f"{i}. {title}")
                        
                        # 새 탭에서 기사 본문 추출
                        article_page = await browser.new_page()
                        try:
                            await article_page.goto(full_url)
                            await asyncio.sleep(2)
                            
                            # 본문 추출
                            article_xpath = '//*[@id="view"]/div[2]/div[3]/article'
                            article_content = await article_page.locator(f"xpath={article_xpath}").text_content()
                            
                            if article_content:
                                article_content = article_content.strip()
                                # 본문이 너무 길면 앞부분만
                                content_preview = article_content[:800] + "..." if len(article_content) > 800 else article_content
                            else:
                                content_preview = "본문을 추출할 수 없습니다."
                            
                            output_list.append({
                                'number': i,
                                'title': title,
                                'url': full_url,
                                'content': content_preview
                            })
                            
                            
                            
                        except Exception as e:
                            print(f"   ❌ 본문 추출 실패: {e}")
                            output_list.append({
                                'number': i,
                                'title': title,
                                'url': full_url,
                                'content': '본문 추출 실패'
                            })
                        finally:
                            await article_page.close()
                
                except Exception as e:
                    print(f"{i}. ❌ 기사 처리 실패: {e}")
                    continue
            
            await browser.close()
            
            if output_list:
                return {"search_results": output_list}

            else:
                return f"'{query}' 기사 추출에 실패했습니다."
                
    except Exception as e:
        return f"스크래핑 중 오류 발생: {str(e)}"


####################################################### 워크플로우 구성 #################################################################


# 워크플로우 구성
workflow = StateGraph(State)

# 노드 추가
workflow.add_node("extract_keyword", extract_keyword)
workflow.add_node("query_expansion", query_expansion)
workflow.add_node("search_news", search_news)
workflow.add_node("generate_response", generate_response)

# 엣지 연결
workflow.add_edge(START, "extract_keyword")
workflow.add_edge("extract_keyword", "query_expansion")
workflow.add_edge("query_expansion", "search_news")
workflow.add_edge("search_news", "generate_response")
workflow.add_edge("generate_response", END)

# 그래프 컴파일
graph = workflow.compile()

#############################################################

# LangGraph의 astream() 메서드를 사용하여 비동기적으로 그래프를 실행하고 결과를 스트리밍

async def async_stream(query):
    # 비동기 스트리밍 함수
    async for event in graph.astream({"query": query}, debug=True):
        yield event


# async_stream이라는 비동기 제너레이터를 실행하고 그 결과를 모아서 반환하는 역할
# syncio.new_event_loop()를 호출하여 새로운 asyncio 이벤트 루프를 생성
# Streamlit과 같이 이미 실행 중인 이벤트 루프가 있는 환경에서 asyncio.run()을 직접 사용했을 때 발생하는 RuntimeError를 피하기 위한 방법
# async def gather_events(): 비동기 제너레이터 gen을 반복적으로 실행하여 모든 이벤트를 events 리스트에 수집하는 역할


def run_async_stream(query):
    # Streamlit 동기 함수에서 비동기 제너레이터를 동기적으로 실행하는 헬퍼
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    gen = async_stream(query)
    
    async def gather_events():
        events = []
        async for event in gen:
            events.append(event)
        return events

    return loop.run_until_complete(gather_events())




if query := st.chat_input("검색할 키워드를 입력하세요"):
    st.subheader(f"🔍 검색: {query}")
    st.subheader("🤖 답변")
    with st.spinner("답변 생성중..."):
        events = run_async_stream(query)

        # 받은 이벤트들을 화면에 출력
        for event in events:
            for k, v in event.items():
                if k == 'extract_keyword':
                    with st.container():
                        st.write("### 🔑 추출된 핵심 키워드")
                        st.markdown(f"**원본 질문:** {query}")
                        st.markdown(f"**추출된 키워드:** {v['extracted_keyword']}")
                        
                elif k == 'query_expansion':
                    with st.container():
                        st.write("### 🔍 확장된 쿼리 리스트")
                        expanded_query_md = '\n'.join([f"- {q}" for q in v['expanded_keywords']])
                        st.markdown(expanded_query_md)
                        
                elif k == 'search_news':
                    with st.expander("📰 검색된 뉴스 (EconoTimes)"):
                        for search_item in v['search_results']:
                            with st.container():
                                st.markdown(f"**제목:** {search_item['title']}")
                                st.markdown(f"**검색 쿼리:** {search_item['search_query']}")
                                st.markdown(f"**URL:** [{search_item['url']}]({search_item['url']})")
                                st.markdown(f"**내용:** {search_item['content'][:500]}...")
                                        
                                st.markdown("---")
                                    
                       
                elif k == 'generate_response':
                    st.markdown("## 📋 최종 분석 리포트")
                    st.markdown(v['answer'])
