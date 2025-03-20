"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: defunct_run_gpt_prompt.py
Description: Defines all run gpt prompt functions using Amazon Bedrock.
"""
import re
import datetime
import sys
import json
import boto3
import os
sys.path.append('../../')

from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *

def get_bedrock_client():
    """Creates and returns a Bedrock client"""
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )
    return bedrock

def invoke_bedrock(prompt, model_params):
    """
    Generic function to invoke Bedrock with given parameters
    """
    try:
        client = get_bedrock_client()
        
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": model_params.get("max_tokens", 2000),
            "temperature": model_params.get("temperature", 0),
            "top_p": model_params.get("top_p", 1),
            "stop_sequences": model_params.get("stop", ["\n"]) if model_params.get("stop") else None
        })
        
        response = client.invoke_model(
            modelId="anthropic.claude-v2",
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion', '')
        
    except Exception as e:
        print(f"Bedrock API Error: {e}")
        return None

def safe_generate_response(prompt, gpt_param, max_retries, fail_safe,
                         validate_func, clean_up_func):
    """
    Safely generate a response using Bedrock with retries and validation
    """
    for i in range(max_retries):
        try:
            response = invoke_bedrock(prompt, gpt_param)
            if response and validate_func(response, prompt):
                return clean_up_func(response, prompt)
        except Exception as e:
            print(f"Error in attempt {i+1}: {e}")
            continue
    return fail_safe

def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False):
    """
    Given the persona, returns an integer that indicates the hour when the 
    persona wakes up.
    """
    def create_prompt_input(persona, test_input=None):
        if test_input: return test_input
        prompt_input = [persona.scratch.get_str_iss(),
                       persona.scratch.get_str_lifestyle(),
                       persona.scratch.get_str_firstname()]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = int(gpt_response.strip().lower().split("am")[0])
        return cr
    
    def __func_validate(gpt_response, prompt=""): 
        try: __func_clean_up(gpt_response, prompt="")
        except: return False
        return True

    def get_fail_safe(): 
        fs = 8
        return fs

    gpt_param = {
        "max_tokens": 5,
        "temperature": 0.8,
        "top_p": 1,
        "stop": ["\n"]
    }
    
    prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
    prompt_input = create_prompt_input(persona, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                  __func_validate, __func_clean_up)
    
    if debug or verbose: 
        print_run_prompts(prompt_template, persona, gpt_param, 
                         prompt_input, prompt, output)
        
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

def run_gpt_prompt_daily_plan(persona, wake_up_hour, test_input=None, verbose=False):
    """
    Creates a daily plan using Bedrock
    """
    def create_prompt_input(persona, wake_up_hour, test_input=None):
        if test_input: return test_input
        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 am"]
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        cr = []
        _cr = gpt_response.split(")")
        for i in _cr:
            if i[-1].isdigit():
                i = i[:-1].strip()
                if i[-1] == "." or i[-1] == ",":
                    cr += [i[:-1].strip()]
        return cr

    def __func_validate(gpt_response, prompt=""):
        try: __func_clean_up(gpt_response, prompt="")
        except: return False
        return True

    def get_fail_safe():
        fs = ['wake up and complete the morning routine at 6:00 am',
              'eat breakfast at 7:00 am',
              'read a book from 8:00 am to 12:00 pm',
              'have lunch at 12:00 pm',
              'take a nap from 1:00 pm to 4:00 pm',
              'relax and watch TV from 7:00 pm to 8:00 pm',
              'go to bed at 11:00 pm']
        return fs

    gpt_param = {
        "max_tokens": 500,
        "temperature": 1,
        "top_p": 1
    }
    
    prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()
    
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                  __func_validate, __func_clean_up)
    output = ([f"wake up and complete the morning routine at {wake_up_hour}:00 am"]
              + output)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                         prompt_input, prompt, output)
        
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
