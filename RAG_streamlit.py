import os  # 파일 작업을 위한 라이브러리
import tempfile  # 임시 파일 저장을 위한 라이브러리
import streamlit as st  # Streamlit을 사용해 웹 앱 구축
from langchain.chat_models import ChatOpenAI  # OpenAI 채팅 모델 사용, + pip install --upgrade langchain openai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader  # PDF 문서 로드
from langchain.memory import ConversationBufferMemory  # 대화 기록 관리를 위한 메모리
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory  # Streamlit에서 메시지 저장
from langchain.embeddings import HuggingFaceEmbeddings  # 텍스트 임베딩 (Hugging Face 모델 사용)
from langchain.callbacks.base import BaseCallbackHandler  # 체인의 작업 중 콜백 핸들링
from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인
from langchain.vectorstores import DocArrayInMemorySearch  # 메모리 내 문서 검색
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 문서를 작은 조각으로 나누기



st.set_page_config(page_title="My Cowork ChatBot", page_icon="🦜")  # 페이지 제목과 아이콘 설정
st.title("LANGCHAIN : My Cowork ChatBot with Documents")  # 페이지 제목 표시

def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200) 
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # all-MiniLM-L6-v2
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectordb.as_retriever(search_type = "mmr", search_kwargs = {"k": 3, "fetch_k": 4})
    # mmr : maximal marginal relevance, 검색의 다양성과 관련성을 동시에 고려해 결과를 반환하는 검색전략
    # k = 2 : 최종적으로 반환할 검색 결과의 수
    # fetch_k : 내부적으로 검색할 문서의 수 설정.
    # 이 값을 통해 검색 효율성과 정확성 간의 균형을 조정.
    
    return retriever

# llm 응답을 실시간 스트리밍 출력하는 핸들러
# llm 실행시작 및 토큰생성마다 콜백을 호출해 ui를 즉각적으로 업데이트함
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text # 컨테이너 안에 표기할 초기텍스트
        self.run_id_ignore_token = None
        
    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts[0].startswith("Human"): # 입력한 질문 출력하지 않도록 수행
            self.run_id_ignore_token = kwargs.get("run_id") # 실행 id를 저장해 특정 실행 무시
            
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token 
        self.container.markdown(self.text) # streamlit 컨테이너에 텍스트 업뎃
        
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")
        
    def on_retriever_start(self, serialized: dict, query : str, **kwargs):
        self.status.write(f"**Question:** {query}") # 처리중인 질문 표시
        self.status.update(label = f"**Context Retrieval:** {query}") # 상태레이블 업뎃
        
        
    def on_retriever_end(self, documents):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
            
        self.status.update(state = "complete") 
        
# 
# api_key = 'AIzaSyDNpp0KaAXNzGQLDTYCopXOKd8lwT7FluI'
# os.environ["GOOGLE_API_KEY"] = api_key 

api_key = st.sidebar.text_input("Google API Key", type="password")

if not api_key:
    st.info("Please add your API key to continue")
    st.stop()
    
uploaded_files = st.sidebar.file_uploader(
    label = "Upload PDF files", type = ["pdf"], accept_multiple_files = True
)

if not uploaded_files:
    st.info("Please upload PDF documents to continue")
    st.stop()
    
# 업로드된 문서로 문서 검색기 설정
retriever = configure_retriever(uploaded_files)

msgs = StreamlitChatMessageHistory()
# 메모리를 사욯애 대화의 문맥을 유지하는 함수. 이전 내용을 기반으로 모델이 적절한 응답 생성하도록 도와줌
memory = ConversationBufferMemory(memory_key = "chat_history", chat_memory = msgs, return_messages = True)
# return_messages : true라면 저자왼 대화 내용을 메시지 객체로 반환

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0.05, google_api_key = api_key)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
) 

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("무엇이 궁금하신가요?") # 기본 메시지 추가
    
# 채팅 인터페이스에 해당 아바타로 메시지 표시하도록
avatars = {"human" : "user", "ai": "assistant"}

# 메시지 기록 반복하며 해당 아바타로 메시지 표시
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)
    
if user_query := st.chat_input(placeholder = "무엇이든 물어봐주세요!"): # 사용자입 입력대기
    st.chat_message("user").write(user_query) # 사용자 질문 표시
    
    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks = [retrieval_handler, stream_handler]) 