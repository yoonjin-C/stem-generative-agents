"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling Amazon Bedrock APIs.
"""
import json
import random
import time 
import boto3

from utils import *

bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def Bedrock_single_request(prompt):
  temp_sleep()

  try:
      completion = bedrock.invoke_model(            
          modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
          body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
          })
     )
      response_body = json.loads(completion.get('body').read())
      return response_body.get('content')[0].get('text', '')
        
  except Exception as e:
      print(f"Error in Bedrock request: {e}")
      return ""

# ============================================================================
# #####################[SECTION 1: BEDROCK STRUCTURE] ######################
# ============================================================================

def Bedrock4_request(prompt): 
  """
  Given a prompt and a dictionary of parameters, make a request to Bedrock
  server and returns the response. 
  ARGS:
    prompt: a str prompt
  RETURNS: 
    a str response from the model. 
  """
  temp_sleep()

  try:
    completion = bedrock.invoke_model(            
          modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
          body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
          })
    )
    response_body = json.loads(completion.get('body').read())
    return response_body.get('content')[0].get('text', '')
  
  except:
    print ("Bedrock ERROR") 
    return "Bedrock ERROR"

def Bedrock_request(prompt): 
  """
  Given a prompt and a dictionary of parameters, make a request to Bedrock
  server and returns the response. 
  ARGS:
    prompt: a str prompt
  RETURNS: 
    a str response from the model. 
  """
  temp_sleep()
  try:
    completion = bedrock.invoke_model(            
          modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
          body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
          })
    )
    response_body = json.loads(completion.get('body').read())
    return response_body.get('content')[0].get('text', '')
  
  except:
    print ("Bedrock ERROR") 
    return "Bedrock ERROR"


def Bedrock4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = f"\n\nBedrock Prompt:\n'''\n{prompt}\n'''\n"
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += f'{{"output": "{str(example_output)}"}}'
  if verbose: 
    print ("Bedrock PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_bedrock_response = Bedrock4_request(prompt).strip()
      end_index = curr_bedrock_response.rfind('}') + 1
      curr_bedrock_response = curr_bedrock_response[:end_index]
      curr_bedrock_response = json.loads(curr_bedrock_response)["output"]
      
      if func_validate(curr_bedrock_response, prompt=prompt): 
        return func_clean_up(curr_bedrock_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_bedrock_response)
        print (curr_bedrock_response)
        print ("~~~~")

    except: 
      pass

  return False


def Bedrock_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = f"\n\nBedrock Prompt:\n'''\n{prompt}\n'''\n"
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += f'{{"output": "{str(example_output)}"}}'
  if verbose: 
    print ("Bedrock PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_bedrock_response = Bedrock_request(prompt).strip()
      end_index = curr_bedrock_response.rfind('}') + 1
      curr_bedrock_response = curr_bedrock_response[:end_index]
      curr_bedrock_response = json.loads(curr_bedrock_response)["output"]
      
      if func_validate(curr_bedrock_response, prompt=prompt): 
        return func_clean_up(curr_bedrock_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_bedrock_response)
        print (curr_bedrock_response)
        print ("~~~~")

    except: 
      pass

  return False


def Bedrock_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("Bedrock PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_bedrock_response = Bedrock_request(prompt).strip()
      if func_validate(curr_bedrock_response, prompt=prompt): 
        return func_clean_up(curr_bedrock_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_bedrock_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: BEDROCK STRUCTURE] ###################
# ============================================================================

def Bedrock_request(prompt, bedrock_parameter): 
  """
  Given a prompt and a dictionary of Bedrock parameters, make a request to Bedrock
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    bedrock_parameter: a python dictionary with the parameter values   
  RETURNS: 
    a str response from the model. 
  """
  temp_sleep()
  try: 
    completion = bedrock.invoke_model(
      modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
      body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": bedrock_parameter["max_tokens"],
        "temperature": bedrock_parameter["temperature"],
        "top_p": bedrock_parameter["top_p"],
        "stop_sequences": bedrock_parameter["stop"] if "stop" in bedrock_parameter else ["\n"],
        "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
      })
    )
    response_body = json.loads(completion.get('body').read())
    return response_body.get('content')[0].get('text', '')
  except Exception as e:
    print(f"Error occurred: {str(e)}")
    return f"ERROR: {str(e)}"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the model. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to the model.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           bedrock_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_bedrock_response = Bedrock_request(prompt, bedrock_parameter)
    if func_validate(curr_bedrock_response, prompt=prompt): 
      return func_clean_up(curr_bedrock_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_bedrock_response)
      print (curr_bedrock_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="amazon.titan-embed-text-v1"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  response = bedrock.invoke_model(
    modelId=model,
    body=json.dumps({
      "inputText": text
    })
  )
  return json.loads(response.get('body').read())['embedding']


if __name__ == '__main__':
  bedrock_parameter = {"max_tokens": 50, 
                   "temperature": 0, "top_p": 1,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(bedrock_response): 
    if len(bedrock_response.strip()) <= 1:
      return False
    if len(bedrock_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(bedrock_response):
    cleaned_response = bedrock_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 bedrock_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















