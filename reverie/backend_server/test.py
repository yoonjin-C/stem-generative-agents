"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import time
import requests  # Bedrock API 호출을 위한 requests 모듈 추가

from utils import *

# Bedrock API 설정
bedrock_api_url = "https://api.bedrock.aws.amazon.com"  # Bedrock API 엔드포인트
bedrock_api_key = "YOUR_BEDROCK_API_KEY"  # Bedrock API 키

def ChatGPT_request(prompt): 
    """
    Given a prompt, make a request to Amazon Bedrock server and returns the response. 
    ARGS:
        prompt: a str prompt
    RETURNS: 
        a str of Bedrock's response. 
    """
    headers = {
        "Authorization": f"Bearer {bedrock_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "input": prompt,
        "model": "your-model-name",  # 사용할 모델 이름으로 변경
        "parameters": {
            "max_tokens": 150,  # 필요한 경우 조정
            "temperature": 0.7  # 필요한 경우 조정
        }
    }
    
    try: 
        response = requests.post(f"{bedrock_api_url}/v1/models/your-model-name/generate", 
                                 headers=headers, 
                                 json=data)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response.json()["output"]  # Bedrock의 응답에서 필요한 데이터 추출
    
    except Exception as e: 
        print("Bedrock API ERROR:", e)
        return "Bedrock API ERROR"

prompt = """
---
Character 1: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, Maria Lopez and Klaus Mueller were already conversing about conversing about Maria's research paper mentioned by Klaus This context takes place after that conversation.

Current Context: Maria Lopez was attending her Physics class (preparing for the next lecture) when Maria Lopez saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Maria Lopez is thinking of initating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Maria Lopez's head: Maria Lopez should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Maria Lopez doesn't necessarily know anything more about Klaus Mueller) 

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Maria Lopez about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Maria Lopez) 

Here is their conversation. 

Maria Lopez: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""

print (ChatGPT_request(prompt))












