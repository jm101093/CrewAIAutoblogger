from crewai_tools import BaseTool, FileReadTool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults, BraveSearch
from langchain_community.utilities import brave_search
from interpreter import interpreter
import time

print("Custom tool module loaded")

log_file_path = 'path/to/your/logfile.log'
file_read_tool = FileReadTool(file_path=log_file_path)

# Define your API key
BraveApi_Key = "Your key"

# Initialize the search tools
search_tool = DuckDuckGoSearchRun()
search_tool1 = DuckDuckGoSearchResults()

# Initialize BraveSearch
print("Initializing BraveSearch")
#brave_search = BraveSearch(api_key=BraveApi_Key)
print("BraveSearch initialized")

class DuckDuckGoSearch(BaseTool):
    name: str = "DuckDuckGoSearch"
    description: str = (
        "Search the web for information on a given topic using DuckDuckGo."
    )
    def _run(self, search_query: str) -> str:
        return search_tool.run(search_query)

class DuckDuckGoSearchResults(BaseTool):
    name: str = "DuckDuckGoSearchResults"
    description: str = (
        "Search the web for information on a given topic using DuckDuckGo and returns results."
    )
    def _run(self, search_query: str) -> str:
        return search_tool1.run(search_query)

class CLIExecutor(BaseTool):
    name: str = "CLIExecutor"
    description: str = (
        "Create and execute CLI commands using Open Interpreter."
    )

    def _run(self, command: str) -> str:
        result = interpreter.chat(command)
        return result

# class LogFileReader(BaseTool):
#     name = "logstream"
#     description: str = (
#         "continuously read the log file"
#     )
#     # Initialize the FileReadTool with the path to your log file
#     # Function to continuously read the log file
#     def continuously_read_log():
#         last_position = 0
#         while True:
#             # Read the current content of the log file
#             content = file_read_tool.read_file(last_position)
#             if content:
#                 print(content)  # Process the new content as needed
#                 last_position += len(content)  # Update the last position
#             time.sleep(1)  # Wait for a second before checking again

#     # Start reading the log file
#     continuously_read_log()


# class BraveSearchTool(BaseTool,BraveApi_Key):
#     name: str = "BraveSearch"
#     description: str = (
#     """
#     Performs a search using BraveSearch API.

#     Parameters:
#     - BraveApi_Key (str): The API key for accessing the BraveSearch API.
#     - search_query (str): The query string for the search.

#     Returns:
#     - Search results from BraveSearch.
#     """
#     )
#     Api_Key = "Your key"

#     def _run(self, Api_Key, search_query: str) -> str:
#         return self.bravesearch.run(search_query)