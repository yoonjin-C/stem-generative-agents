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

bedrock = boto3.client('bedrock-runtime')

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt):
  temp_sleep()

  completion = bedrock.invoke_model(
    modelId="anthropic.claude-v2",
    body=json.dumps({
      "prompt": prompt,
      "max_tokens_to_sample": 2048,
      "temperature": 0
    })
  )
  response = json.loads(completion.get('body').read())
  return response['completion']

# ============================================================================
# #####################[SECTION 1: BEDROCK STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
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
      modelId="anthropic.claude-v2",
      body=json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 2048,
        "temperature": 0
      })
    )
    response = json.loads(completion.get('body').read())
    return response['completion']
  
  except:
    print ("Bedrock ERROR") 
    return "Bedrock ERROR"

def ChatGPT_request(prompt): 
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
      modelId="anthropic.claude-v2", 
      body=json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 2048,
        "temperature": 0
      })
    )
    response = json.loads(completion.get('body').read())
    return response['completion']
  
  except:
    print ("Bedrock ERROR") 
    return "Bedrock ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_response = GPT4_request(prompt).strip()
      end_index = curr_response.rfind('}') + 1
      curr_response = curr_response[:end_index]
      curr_response = json.loads(curr_response)["output"]
      
      if func_validate(curr_response, prompt=prompt): 
        return func_clean_up(curr_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_response)
        print (curr_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_response = ChatGPT_request(prompt).strip()
      end_index = curr_response.rfind('}') + 1
      curr_response = curr_response[:end_index]
      curr_response = json.loads(curr_response)["output"]
      
      if func_validate(curr_response, prompt=prompt): 
        return func_clean_up(curr_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_response)
        print (curr_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_response, prompt=prompt): 
        return func_clean_up(curr_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: BEDROCK STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of parameters, make a request to Bedrock
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the parameter values   
  RETURNS: 
    a str response from the model. 
  """
  temp_sleep()
  try: 
    completion = bedrock.invoke_model(
      modelId="anthropic.claude-v2",
      body=json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": gpt_parameter["max_tokens"],
        "temperature": gpt_parameter["temperature"],
        "top_p": gpt_parameter["top_p"],
        "stop_sequences": gpt_parameter["stop"] if "stop" in gpt_parameter else None
      })
    )
    response = json.loads(completion.get('body').read())
    return response['completion']
  except: 
    print ("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


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
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_response, prompt=prompt): 
      return func_clean_up(curr_response, prompt=prompt)
    if verbose: 
      print ("---- repeat count: ", i, curr_response)
      print (curr_response)
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
  gpt_parameter = {"max_tokens": 50, 
                   "temperature": 0, "top_p": 1,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(response): 
    if len(response.strip()) <= 1:
      return False
    if len(response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(response):
    cleaned_response = response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















