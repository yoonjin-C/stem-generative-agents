"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: print_prompt.py
Description: For printing prompts when the setting for verbose is set to True.
"""
import sys
sys.path.append('../')

import json
import numpy
import datetime
import random
import boto3

from global_methods import *
from persona.prompt_template.gpt_structure import *
from utils import *
from test import *

##############################################################################
#                    PERSONA Chapter 1: Prompt Structures                      #
##############################################################################

def print_run_prompts(prompt_template=None, 
                     persona=None, 
                     bedrock_param=None, 
                     prompt_input=None,
                     prompt=None, 
                     output=None): 
    """
    Prints the prompts and responses in a formatted way.
    Now supports both GPT and Bedrock outputs.
    """
    print(f"=== {prompt_template}")
    print("~~~ persona    ---------------------------------------------------")
    print(persona.name, "\n")
    print("~~~ bedrock_param ----------------------------------------------------")
    print(bedrock_param, "\n")
    print("~~~ prompt_input    ----------------------------------------------")
    print(prompt_input, "\n")
    print("~~~ prompt    ----------------------------------------------------")
    print(prompt, "\n")
    print("~~~ output    ----------------------------------------------------")
    print(output, "\n") 
    print("=== END ==========================================================")
    print("\n\n\n")


    # 래퍼(wrapper) 함수

# def format_bedrock_response(response):
#     """
#     Helper function to format Bedrock response to match expected output format
#     """
#     try:
#         # If the response is already in the expected format, return as is
#         if isinstance(response, (list, dict)):
#             return response
        
#         # Try to parse as JSON if it's a string
#         if isinstance(response, str):
#             try:
#                 return json.loads(response)
#             except json.JSONDecodeError:
#                 # If not JSON, return as is
#                 return response
                
#     except Exception as e:
#         print(f"Error formatting Bedrock response: {e}")
#         return response

# def run_gpt_prompt(prompt, bedrock_param=None):
#     """
#     Replacement for GPT prompt runner that uses Bedrock instead.
#     Maintains similar interface to minimize impact on existing code.
#     """
#     if bedrock_param is None:
#         bedrock_param = {
#             "temperature": 0.7,
#             "max_tokens": 2000
#         }
    
#     response = Bedrock_request(
#         prompt,
#         # temperature=bedrock_param.get("temperature", 0.7),
#         # max_tokens=bedrock_param.get("max_tokens", 2000)
#     )
    
#     return format_bedrock_response(response)

# If you have any other GPT-specific functions, add their Bedrock equivalents here
