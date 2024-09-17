from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any
from crewai import Agent, Crew, Process, Task, tools
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
import asyncio
import json

# Load environment variables from .env file
load_dotenv()

# Must precede any llm module imports
langtrace.init(api_key='Your key')

# Initialize search tools
s1 = DuckDuckGoSearch()
s2 = DuckDuckGoSearchResults()
s3 = WebsiteSearchTool()
s4 = EXASearchTool()
s5 = InternetSearchTool()  # serper

# Set environment variables
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-Your key-MUFOd8FHzbeBobvkvPtA2giLxQ-6CD-ZZw-ZSNnUAAA"

# LLM configurations
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
    model_name="gemma-7b-it"
)

llm_groq3 = ChatGroq(
    temperature=0, 
    groq_api_key=os.getenv("Groq_api_key"), 
    model_name="gemma2-9b-it"
)

llm_groq4 = ChatGroq(
    temperature=0, 
    groq_api_key=os.getenv("Groq_api_key"), 
    model_name="llama-3.1-70b-versatile"
)
# FastAPI app setup
app = FastAPI()

# Mount a static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

class BlogInput(BaseModel):
    location: str
    year: str

class BlogOutput(BaseModel):
    content: str

class CustomCrew(Crew):
    async def log_message(self, message: str):
        if hasattr(self, 'log_queue'):
            await self.log_queue.put(message)

@CrewBase
class FirstnewCrew():
    """Firstnew crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def idea_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['idea_generator'],
            tools=[s1, s2, s3, s4, s5],
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
            llm=llm_googleflash,
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
            llm=llm_google,
            max_iter=800,
            memory=True,
            max_rpm=15
        )

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

    @crew
    def crew(self) -> CustomCrew:
        return CustomCrew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "openai",
                "config": {
                    "model": "nomic-embed-text-v1.5-f16"
                }
            },
        )

    async def run_crew(self, inputs: dict, log_queue: asyncio.Queue) -> str:
        crew_instance = self.crew()
        crew_instance.log_queue = log_queue
        result = await asyncio.to_thread(crew_instance.kickoff, inputs=inputs)
        
        try:
            with open('report.md', 'r') as file:
                report_content = file.read()
            return report_content
        except FileNotFoundError:
            return "Report file not found. The crew may not have generated a report."
        except Exception as e:
            return f"An error occurred while reading the report: {str(e)}"

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/write_blog/")
async def write_blog(location: str, year: str):
    async def event_stream():
        log_queue = asyncio.Queue()
        inputs = {
            'Location': location,
            'year': year
        }
        
        try:
            crew = FirstnewCrew()
            content_task = asyncio.create_task(crew.run_crew(inputs, log_queue))
            
            while True:
                try:
                    message = await asyncio.wait_for(log_queue.get(), timeout=0.1)
                    yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"
                except asyncio.TimeoutError:
                    if content_task.done():
                        break
            
            content = await content_task
            yield f"data: {json.dumps({'type': 'result', 'content': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)