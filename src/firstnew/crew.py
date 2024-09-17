from crewai import Agent, Crew, Process, Task, tools, Process
from crewai_tools import ScrapeWebsiteTool, WebsiteSearchTool, EXASearchTool, DallETool
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from firstnew.tools.custom_tool import DuckDuckGoSearch, DuckDuckGoSearchResults
from langchain_community.llms import Ollama
from firstnew.tools.S_tool import InternetSearchTool
from dotenv import load_dotenv
import os
from langtrace_python_sdk import langtrace
#from firstnew.tools.custom_tool import DuckDuckGoSearch, DuckDuckGoSearchResults, BraveSearchTool
# Load environment variables from .env file
load_dotenv()

# Must precede any llm module imports
langtrace.init(api_key = 'Your key')

s1 = DuckDuckGoSearch()
s2 = DuckDuckGoSearchResults()
s3 = WebsiteSearchTool()
s4 = EXASearchTool()
s5 = InternetSearchTool() #serper

os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["ANTHROPIC_API_KEY"] = "Your key"
#s.environ["CHAT_OLLAMA_BASE_URL"] ="http://localhost:11434/api/chat"
#llms configuration
llm_openai = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1/images/generations",
    openai_api_key="Your key"
)

llm_lmstudio = ChatOpenAI(
    openai_api_key="null",
    openai_api_base="http://hal9000:1234/v1",              
    model_name="llama"
)

llm_mistral = ChatOpenAI(
    openai_api_key="Your key",
    openai_api_base="https://api.mistral.ai/v1",
    model_name="open-mixtral-8x7b"
)

llm_claude = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_ollama = ChatOllama(model="llama3.1:latest")
llm_ollama2 = ChatOllama(model="mistral:latest")
llm_ollama3 = ChatOllama(model="phi3:latest")
llm_ollama4 = ChatOllama(model="dolphin-llama3:8b-256k")
llm_ollama5 = ChatOllama(model="gemma2:latest")

llm_google = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro", verbose=True, temperature=0.5, google_api_key=os.getenv("google_api_key")
)

llm_googleflash = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", verbose=True, temperature=0.5, google_api_key=os.getenv("google_api_key")
)

llm_groq = ChatGroq(
    temperature=0, 
    groq_api_key=os.getenv("Groq_api_key"), 
    model_name="llama3-8b-8192"
)

llm_groq2 = ChatGroq(
    temperature=0, 
    groq_api_key=os.getenv("Groq_api_key"), 
    model_name="llama3-groq-70b-8192-tool-use-preview"
)

@CrewBase
class FirstnewCrew():
    """Firstnew crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def idea_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['idea_generator'],
            tools=[s1, s2, s3, s4,s5],
            allow_delegation=False,
            verbose=True,
            llm=llm_groq,
            max_iter=30,
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[s5, s4, s3, s2, s1],
            allow_delegation=False,
            verbose=True,
            llm=llm_ollama,
            max_iter=500,
            memory=True,
            max_rpm=15
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True,
            allow_delegation=True,
            llm=llm_lmstudio,
            max_iter=800,
            memory=True,
            max_rpm=15
        )
    # @agent
    # def image_generator(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['image_generator'],
    #         verbose=True,
    #         allow_delegation=False,
    #         llm=llm_openai,
    #         tools=[DallETool(model="dall-e-3", size="1024X1024")],
    #         max_iter=50,
    #         memory=True,
    #     )
    @task
    def ideas_task(self) -> Task:
        return Task(
            config=self.tasks_config['ideas_task'],
            agent=self.idea_generator(),
        )
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher(),
            async_execution=True,
            context=[self.ideas_task()]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            agent=self.reporting_analyst(),
            context=[self.research_task(), self.ideas_task()],
            output_file='report.md'
        )
    # @task
    # def image_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['image_task'],
    #         agent=self.image_generator(),
    #         context=[self.research_task(), self.ideas_task(), self.reporting_task()],
    #         output_file='images.txt',
    #     )
    @crew
    def crew(self) -> Crew: #storage_path=None
    
      return Crew(
        agents=self.agents,
        tasks=self.tasks,
        process=Process.sequential,
        verbose=True,
        memory=True,
        #planning=True,
        embedder={
            "provider": "openai",
            "config": {
                "model": "nomic-embed-text-v1.5-f16"
            }
        },
    )