#!/usr/bin/env python
import sys
from firstnew.crew import FirstnewCrew
from textwrap import dedent
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Union

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origens=["*"],
#     allow_credentials=True,
#     allow_meathods=["*"]
#     allow_headers=["*"]
# )
# templates = Jinja2Templates(directory="templates")

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# def run():
#     FirstnewCrew().crew().kickoff()

def run():
    print("## Welcome to the blog Writer")
    print('-------------------------------')
    userInput = input("What is the Location you want to focus on?\n")
    userInput1 = input("What is the current year?\n")
    
    inputs = {
        'Location': userInput,
        'year': userInput1
    }
    FirstnewCrew().crew().kickoff(inputs=inputs)

def train():
    """
    Train the crew for a given number of iterations.
    """
    print("## Welcome to the blog Writer")
    print('-------------------------------')
    userInput = input("What is the Location you want to focus on?\n")
    userInput1 = input("What is the current year?\n")
    
    inputs = {
        'Location': userInput,
        'year': userInput1
    }
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'Location': userInput,
        'year': userInput1
    }
    try:
        FirstnewCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")