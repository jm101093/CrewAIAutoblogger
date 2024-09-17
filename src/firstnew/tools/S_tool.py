import json
import os
import requests
from crewai_tools import BaseTool
from crewai_tools import DallETool
from dotenv import load_dotenv

load_dotenv()

class InternetSearchTool(BaseTool):
    name: str = "InternetSearch"
    description: str = "Search the internet for information on a given topic."

    def _run(self, query: str) -> str:
        """Search the internet about a given topic and return relevant results."""
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'content-type': 'application/json'
        }
        response = requests.post(url, headers=headers, data=payload)
        
        if 'organic' not in response.json():
            return "Sorry, I couldn't find anything about that. There could be an error with your Serper API key."
        
        results = response.json().get('organic', [])
        result_strings = []

        for result in results[:top_result_to_return]:
            try:
                result_strings.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Snippet: {result['snippet']}",
                    "\n-----------------"
                ]))
            except KeyError:
                continue

        return '\n'.join(result_strings)

