# **Comprehensive Step-by-Step Guide to Building the Bonzo Conversational Bot Using Pydantic Models**

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Step 1: Set Up the Project Environment](#step-1-set-up-the-project-environment)
5. [Step 2: Define the Bonzo Conversation Model Using Pydantic](#step-2-define-the-bonzo-conversation-model-using-pydantic)
6. [Step 3: Implement Bonzo API Integration](#step-3-implement-bonzo-api-integration)
7. [Step 4: Implement OpenAI Integration](#step-4-implement-openai-integration)
8. [Step 5: Develop the Agents](#step-5-develop-the-agents)
    - [5.1. Orchestrator Agent](#51-orchestrator-agent)
    - [5.2. Conversation Agent](#52-conversation-agent)
    - [5.3. Policy Compliance Agent](#53-policy-compliance-agent)
    - [5.4. Knowledge Agent](#54-knowledge-agent)
9. [Step 6: Implement the Main Application Logic](#step-6-implement-the-main-application-logic)
10. [Step 7: Implement Unit Tests](#step-7-implement-unit-tests)
11. [Step 8: Testing the Application](#step-8-testing-the-application)
12. [Step 9: Deployment Considerations](#step-9-deployment-considerations)
13. [Conclusion](#conclusion)
14. [Appendix: Sample Conversation Payloads](#appendix-sample-conversation-payloads)

---

## **Project Overview**

We will build a production-ready conversational chatbot for Bonzo CRM that:

- **Receives incoming messages** from prospects, including detailed conversation payloads.
- **Parses and understands** the conversation payloads using Pydantic models.
- **Determines the appropriate action** using an orchestrator agent, considering the conversation context and payload details.
- **Crafts responses** using OpenAI's Language Models (LLMs) tailored for sales qualification needs.
- **Sends SMS messages and adds notes** to prospects via Bonzo's API.
- **Ensures compliance** with company policies.
- **Logs interactions** for future analysis.

---

## **Prerequisites**

- Python 3.8 or higher installed.
- Accounts and API keys for:
  - **Bonzo API**.
  - **OpenAI API**.
- Basic understanding of Python, RESTful APIs, and data modeling.
- Knowledge of data serialization/deserialization (JSON handling).

---

## **Project Structure**

```
bonzo_chatbot/
├── .env
├── requirements.txt
├── main.py
├── bonzo_api.py
├── openai_api.py
├── models/
│   ├── __init__.py
│   ├── conversation.py
├── agents/
│   ├── __init__.py
│   ├── orchestrator_agent.py
│   ├── conversation_agent.py
│   ├── policy_compliance_agent.py
│   ├── knowledge_agent.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
├── tests/
│   ├── __init__.py
│   ├── test_bonzo_api.py
│   ├── test_agents.py
│   ├── test_models.py
```

---

## **Step 1: Set Up the Project Environment**

### **1.1. Create the Project Directory**

```bash
mkdir bonzo_chatbot
cd bonzo_chatbot
```

### **1.2. Initialize a Virtual Environment**

```bash
python -m venv venv
```

### **1.3. Activate the Virtual Environment**

- On Windows:

  ```bash
  venv\Scripts\activate
  ```

- On macOS/Linux:

  ```bash
  source venv/bin/activate
  ```

### **1.4. Create `requirements.txt`**

Create a `requirements.txt` file with the following content:

```plaintext
openai==0.27.8
requests==2.26.0
python-dotenv==0.19.0
pydantic==1.10.2
Flask==2.0.1
```

### **1.5. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **1.6. Create a `.env` File**

Create a `.env` file in the project root to store environment variables:

```bash
touch .env
```

Add the following content to `.env` (replace placeholders with your actual API keys):

```ini
# Bonzo API Configuration
BONZO_API_KEY=your_bonzo_api_key
BONZO_API_ENDPOINT=https://app.getbonzo.com/api/v3

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key

# Logging Configuration
LOG_LEVEL=INFO
```

**Important:** Never commit your `.env` file to version control. Add it to `.gitignore`:

```bash
echo ".env" >> .gitignore
```

### **1.7. Create Directories**

```bash
mkdir agents
mkdir utils
mkdir models
mkdir tests
```

---

## **Step 2: Define the Bonzo Conversation Model Using Pydantic**

We will use Pydantic to define data models that match the `BonzoConversationModel` provided. This ensures data validation and easy parsing of conversation payloads.

### **2.1. Create `models/conversation.py`**

```bash
touch models/conversation.py
```

### **2.2. Implement the Conversation Data Models**

```python
# models/conversation.py

from pydantic import BaseModel, Field, root_validator, validator
from typing import Optional
from datetime import datetime
import re

class UserDataExtension(BaseModel):
    user_id: Optional[str] = Field(None, description="ID of the user involved in the conversation.")
    avatar_url: Optional[str] = Field(None, description="URL to the avatar image of the user.")

class MediaContentExtension(BaseModel):
    audio_path: Optional[str] = Field(None, description="URL to audio file if available.")
    image_path: Optional[str] = Field(None, description="URL to image file if available.")

class EventPipelineExtension(BaseModel):
    event_id: Optional[int] = Field(None, description="Unique identifier for the associated event in Bonzo.")
    pipeline_event_id: Optional[int] = Field(None, description="ID for the pipeline event related to the conversation.")

class MetadataExtension(BaseModel):
    sequence_id: Optional[int] = Field(None, description="Identifier for the conversation sequence.")
    source: Optional[str] = Field(None, description="Origin of the conversation or update.", regex="^(direct|update|null)$")
    subject: Optional[str] = Field(None, description="Subject of the conversation, usually for emails.")
    error_message: Optional[str] = Field(None, description="Error message if there was an issue with the conversation.")
    send_at: Optional[datetime] = Field(None, description="Scheduled send time for the message.")
    event_date: Optional[datetime] = Field(None, description="Date of the event linked to the conversation.")

class BonzoConversationModel(BaseModel):
    id: int = Field(..., description="Unique identifier for each conversation entry.")
    prospect_id: int = Field(..., description="Unique identifier for the prospect.")
    created_at: datetime = Field(..., description="Timestamp when the conversation occurred.")
    content: str = Field(..., description="Content of the conversation, text, or transcription.")
    type: Optional[str] = Field(None, description="Type of message.", regex="^(sms|call|email|note|null)$")
    direction: Optional[str] = Field(None, description="Direction of the communication.", regex="^(incoming|outgoing|null)$")
    status: Optional[str] = Field(None, description="Status of the content.", regex="^(delivered|not_delivered|unanswered|answered|cancelled|opened|null)$")
    user_name: Optional[str] = Field(None, description="Name of the user involved.")

    # Extensions
    user_data_extension: Optional[UserDataExtension] = None
    media_content_extension: Optional[MediaContentExtension] = None
    event_pipeline_extension: Optional[EventPipelineExtension] = None
    metadata_extension: Optional[MetadataExtension] = None

    @root_validator(pre=True)
    def apply_extension_rules(cls, values):
        type_ = values.get('type')
        source = values.get('source')
        event_id = values.get('event_id')

        # Apply extension rules based on 'type'
        if type_ == 'call':
            values['user_data_extension'] = UserDataExtension(**values.get('user_data_extension', {}))
            values['media_content_extension'] = MediaContentExtension(**values.get('media_content_extension', {}))
        elif type_ == 'email':
            values['user_data_extension'] = UserDataExtension(**values.get('user_data_extension', {}))
            values['metadata_extension'] = MetadataExtension(**values.get('metadata_extension', {}))
        elif type_ == 'sms':
            values['user_data_extension'] = UserDataExtension(**values.get('user_data_extension', {}))
            values['metadata_extension'] = MetadataExtension(**values.get('metadata_extension', {}))
        elif type_ == 'note':
            values['user_data_extension'] = UserDataExtension(**values.get('user_data_extension', {}))

        # Apply extension rules based on 'source'
        if source == 'update':
            values['metadata_extension'] = MetadataExtension(**values.get('metadata_extension', {}))

        # Apply extension rules based on 'event_id'
        if event_id is not None:
            values['event_pipeline_extension'] = EventPipelineExtension(**values.get('event_pipeline_extension', {}))

        return values
```

### **2.3. Example Usage**

```python
# Example of parsing a conversation payload

from models.conversation import BonzoConversationModel

def parse_conversation_payload(payload):
    conversation = BonzoConversationModel(**payload)
    return conversation
```

---

## **Step 3: Implement Bonzo API Integration**

We'll update `bonzo_api.py` to handle conversation payloads and include functions to fetch and parse conversation history using the Pydantic models.

### **3.1. Update `bonzo_api.py`**

```python
# bonzo_api.py

import os
import requests
from dotenv import load_dotenv
from models.conversation import BonzoConversationModel
from typing import List
import logging

load_dotenv()
logger = logging.getLogger(__name__)

BONZO_API_KEY = os.getenv('BONZO_API_KEY')
BONZO_API_ENDPOINT = os.getenv('BONZO_API_ENDPOINT')

headers = {
    "Authorization": f"Bearer {BONZO_API_KEY}",
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "application/json",
}

def send_sms(prospect_id, message, attachment_url=None):
    url = f"{BONZO_API_ENDPOINT}/prospects/{prospect_id}/sms"
    data = {
        "message": message
    }
    if attachment_url:
        data["attachment_url"] = attachment_url
    response = requests.post(url, headers=headers, data=data)
    return response

def add_note(prospect_id, content, is_pinned=False, include_in_conversation=False):
    url = f"{BONZO_API_ENDPOINT}/prospects/{prospect_id}/notes"
    data = {
        "content": content,
        "is_pinned": str(is_pinned).lower(),
        "include_in_conversation": str(include_in_conversation).lower()
    }
    response = requests.post(url, headers=headers, data=data)
    return response

def get_conversation_history(prospect_id, per_page=15, page=1) -> List[BonzoConversationModel]:
    url = f"{BONZO_API_ENDPOINT}/conversations"
    params = {
        "search": prospect_id,
        "per_page": per_page,
        "page": page,
        "include_conversationlist": True,
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        conversations_data = response.json().get('data', [])
        conversations = []
        for convo_data in conversations_data:
            try:
                conversation = BonzoConversationModel(**convo_data)
                conversations.append(conversation)
            except Exception as e:
                logger.error(f"Error parsing conversation: {e}")
        return conversations
    else:
        logger.error(f"Failed to fetch conversation history: {response.text}")
        return []
```

### **3.2. Test Bonzo API Functions**

Ensure that your Bonzo API key is valid and test the functions.

---

## **Step 4: Implement OpenAI Integration**

Update `openai_api.py` as needed.

### **4.1. Update `openai_api.py`**

```python
# openai_api.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
```

---

## **Step 5: Develop the Agents**

We'll develop agents with prompts tailored for sales qualification needs, filling in gaps where necessary.

### **5.1. Orchestrator Agent**

#### **5.1.1. Update `agents/orchestrator_agent.py`**

```python
# agents/orchestrator_agent.py

from openai_api import openai
import json

def orchestrator_agent(conversation_history, user_input):
    system_prompt = """
You are an orchestrator agent for a sales chatbot. Based on the conversation history and the latest user input, determine the next action.

Actions:
- 'knowledge_retrieval': The user asks a question that requires additional information.
- 'policy_compliance': The drafted response needs to be checked against company policies.
- 'simple_response': The bot can respond directly without additional checks.

Provide your decision in the following JSON format:
{
    "action": "chosen_action",
    "reasoning": "Brief explanation of your choice"
}
"""

    # Convert conversation history to text
    conversation_history_text = ""
    for convo in conversation_history:
        sender = "Prospect" if convo.direction == "incoming" else "Assistant"
        timestamp = convo.created_at.strftime("%Y-%m-%d %H:%M:%S")
        message = f"{sender} ({timestamp}): {convo.content}"
        conversation_history_text += message + "\n"

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": conversation_history_text},
            {"role": "user", "content": f"User Input: {user_input}"}
        ],
        functions=[
            {
                "name": "decide_next_action",
                "description": "Decide the next action for the chatbot.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["knowledge_retrieval", "policy_compliance", "simple_response"]
                        },
                        "reasoning": {"type": "string"}
                    },
                    "required": ["action", "reasoning"]
                }
            }
        ],
        function_call={"name": "decide_next_action"}
    )

    decision = response["choices"][0]["message"]["function_call"]["arguments"]
    return json.loads(decision)
```

#### **5.1.2. Example Usage**

```python
# In main.py or wherever needed
decision = orchestrator_agent(conversation_history, user_input)
```

### **5.2. Conversation Agent**

#### **5.2.1. Update `agents/conversation_agent.py`**

```python
# agents/conversation_agent.py

from openai_api import openai

def conversation_agent(conversation_history, user_input, knowledge=None):
    system_prompt = """
You are a helpful sales assistant for a CRM system. Your goal is to engage with the prospect to qualify them and guide them towards booking a meeting.

Instructions:
- Use a friendly and professional tone.
- Ask open-ended questions to understand the prospect's needs.
- Provide concise and relevant information.
- Avoid making any unverified claims.
- If you use any knowledge, reference it appropriately.

If you need to incorporate any specific knowledge, it will be provided below.

"""

    if knowledge:
        system_prompt += f"\nKnowledge to incorporate:\n{knowledge}\n"

    # Convert conversation history to text
    conversation_history_text = ""
    for convo in conversation_history:
        sender = "Prospect" if convo.direction == "incoming" else "Assistant"
        timestamp = convo.created_at.strftime("%Y-%m-%d %H:%M:%S")
        message = f"{sender} ({timestamp}): {convo.content}"
        conversation_history_text += message + "\n"

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": conversation_history_text},
            {"role": "user", "content": user_input}
        ]
    )
    drafted_message = response["choices"][0]["message"]["content"].strip()
    return drafted_message
```

### **5.3. Policy Compliance Agent**

#### **5.3.1. Update `agents/policy_compliance_agent.py`**

```python
# agents/policy_compliance_agent.py

from openai_api import openai
import json

def policy_compliance_agent(drafted_message):
    system_prompt = """
You are a compliance officer for a sales organization. Review the following message for compliance with company policies.

Policies:
- Do not make unverified claims.
- Do not use inappropriate language.
- Ensure all statements are truthful and accurate.
- Maintain confidentiality and do not share sensitive information.

Provide your decision in the following JSON format:
{
    "approved": true or false,
    "reasons": "Reasons if not approved"
}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": drafted_message}
        ],
        functions=[
            {
                "name": "compliance_check",
                "description": "Check the message for policy compliance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "approved": {"type": "boolean"},
                        "reasons": {"type": "string"}
                    },
                    "required": ["approved", "reasons"]
                }
            }
        ],
        function_call={"name": "compliance_check"}
    )

    result = response["choices"][0]["message"]["function_call"]["arguments"]
    return json.loads(result)
```

### **5.4. Knowledge Agent**

#### **5.4.1. Update `agents/knowledge_agent.py`**

```python
# agents/knowledge_agent.py

def knowledge_agent(user_input):
    # Implement your knowledge retrieval logic here
    # For demonstration, we'll simulate knowledge retrieval
    # In production, you might query a database or knowledge base

    # Simulated knowledge base
    knowledge_base = {
        "pricing": "Our pricing is flexible and depends on your specific needs. We offer monthly and annual plans.",
        "features": "Our CRM offers contact management, email marketing, and sales pipeline tracking.",
        "integration": "We integrate with popular tools like Slack, Gmail, and Outlook."
    }

    # Simple keyword matching
    knowledge = ""
    for key, value in knowledge_base.items():
        if key in user_input.lower():
            knowledge += f"{value}\n"

    if not knowledge:
        knowledge = "I'm sorry, I don't have that information at the moment."

    return knowledge.strip()
```

---

## **Step 6: Implement the Main Application Logic**

We will update `main.py` to handle the conversation payloads appropriately and coordinate the agents.

### **6.1. Update `main.py`**

```python
# main.py

from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

from bonzo_api import send_sms, add_note, get_conversation_history
from agents.orchestrator_agent import orchestrator_agent
from agents.conversation_agent import conversation_agent
from agents.policy_compliance_agent import policy_compliance_agent
from agents.knowledge_agent import knowledge_agent
from utils.logger import setup_logger
import logging

load_dotenv()
logger = setup_logger()

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_incoming_message():
    data = request.get_json()
    prospect_id = data.get('prospect_id')
    user_input = data.get('message')

    if not prospect_id or not user_input:
        return jsonify({'status': 'error', 'message': 'Prospect ID or message missing'}), 400

    # Retrieve conversation history
    conversation_history = get_conversation_history(prospect_id)

    # Orchestrator decides next action
    decision = orchestrator_agent(conversation_history, user_input)
    action = decision.get('action')
    reasoning = decision.get('reasoning')

    logger.info(f"Orchestrator decision: {action} - {reasoning}")

    knowledge = None
    if action == 'knowledge_retrieval':
        knowledge = knowledge_agent(user_input)

    # Craft response
    drafted_message = conversation_agent(conversation_history, user_input, knowledge)

    # Add note explaining reasoning
    note_content = f"Action: {action}\nReasoning: {reasoning}\nDrafted Message: {drafted_message}"
    note_response = add_note(prospect_id, note_content)
    if note_response.status_code != 200:
        logger.error(f"Failed to add note: {note_response.text}")

    # Policy compliance check if needed
    if action == 'policy_compliance':
        approval_result = policy_compliance_agent(drafted_message)
        if not approval_result['approved']:
            logger.warning(f"Message not approved: {approval_result['reasons']}")
            return jsonify({'status': 'Message rejected by compliance'}), 200

    # Send the message via Bonzo API
    sms_response = send_sms(prospect_id, drafted_message)
    if sms_response.status_code == 200:
        status = 'Message sent'
        logger.info(status)
    else:
        status = f'Failed to send message: {sms_response.text}'
        logger.error(status)

    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

## **Step 7: Implement Unit Tests**

We will create unit tests for the Bonzo API functions, agents, and Pydantic models.

### **7.1. Set Up Testing Framework**

We will use `unittest` for testing.

### **7.2. Create `tests/test_bonzo_api.py`**

```python
# tests/test_bonzo_api.py

import unittest
from bonzo_api import send_sms, add_note, get_conversation_history
from unittest.mock import patch

class TestBonzoAPI(unittest.TestCase):

    @patch('bonzo_api.requests.post')
    def test_send_sms_success(self, mock_post):
        mock_post.return_value.status_code = 200
        response = send_sms('12345', 'Test message')
        self.assertEqual(response.status_code, 200)

    @patch('bonzo_api.requests.post')
    def test_add_note_success(self, mock_post):
        mock_post.return_value.status_code = 200
        response = add_note('12345', 'Test note')
        self.assertEqual(response.status_code, 200)

    @patch('bonzo_api.requests.get')
    def test_get_conversation_history(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {'data': []}
        conversations = get_conversation_history('12345')
        self.assertIsInstance(conversations, list)

if __name__ == '__main__':
    unittest.main()
```

### **7.3. Create `tests/test_agents.py`**

```python
# tests/test_agents.py

import unittest
from agents.orchestrator_agent import orchestrator_agent
from agents.conversation_agent import conversation_agent
from agents.policy_compliance_agent import policy_compliance_agent
from agents.knowledge_agent import knowledge_agent

class TestAgents(unittest.TestCase):

    def test_orchestrator_agent(self):
        conversation_history = []
        user_input = "Can you tell me about your pricing?"
        decision = orchestrator_agent(conversation_history, user_input)
        self.assertIn(decision['action'], ['knowledge_retrieval', 'policy_compliance', 'simple_response'])
        self.assertIn('reasoning', decision)

    def test_conversation_agent(self):
        conversation_history = []
        user_input = "Tell me more about your services."
        response = conversation_agent(conversation_history, user_input)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_policy_compliance_agent(self):
        drafted_message = "Buy now and get 50% off!"
        result = policy_compliance_agent(drafted_message)
        self.assertIn('approved', result)
        self.assertIn('reasons', result)

    def test_knowledge_agent(self):
        user_input = "What features do you offer?"
        knowledge = knowledge_agent(user_input)
        self.assertIsInstance(knowledge, str)

if __name__ == '__main__':
    unittest.main()
```

### **7.4. Create `tests/test_models.py`**

```python
# tests/test_models.py

import unittest
from models.conversation import BonzoConversationModel
from datetime import datetime

class TestBonzoConversationModel(unittest.TestCase):
    def test_valid_conversation(self):
        data = {
            "id": 1,
            "prospect_id": 123,
            "created_at": "2023-08-01T12:00:00Z",
            "content": "Hello!",
            "type": "sms",
            "direction": "incoming",
            "status": "delivered",
            "user_name": "John Doe",
            "user_data_extension": {
                "user_id": "u123",
                "avatar_url": "http://example.com/avatar.png"
            }
        }
        conversation = BonzoConversationModel(**data)
        self.assertEqual(conversation.id, 1)
        self.assertEqual(conversation.type, "sms")

    def test_invalid_type(self):
        data = {
            "id": 1,
            "prospect_id": 123,
            "created_at": "2023-08-01T12:00:00Z",
            "content": "Hello!",
            "type": "invalid_type",
            "direction": "incoming",
            "status": "delivered"
        }
        with self.assertRaises(ValueError):
            BonzoConversationModel(**data)

if __name__ == '__main__':
    unittest.main()
```

### **7.5. Run Unit Tests**

```bash
python -m unittest discover tests
```

---

## **Step 8: Testing the Application**

### **8.1. Run the Application**

```bash
python main.py
```

### **8.2. Simulate Incoming Messages**

Use tools like `curl` or Postman.

**Example using `curl`:**

```bash
curl -X POST http://localhost:8000/webhook \
   -H 'Content-Type: application/json' \
   -d '{"prospect_id": "12345", "message": "I would like to know more about your pricing."}'
```

### **8.3. Check Logs and Responses**

Ensure that the application is processing the message correctly and that the agents are functioning as expected.

---

## **Step 9: Deployment Considerations**

### **9.1. Use a Production WSGI Server**

Consider using Gunicorn or uWSGI.

**Example with Gunicorn:**

```bash
gunicorn main:app -b 0.0.0.0:8000
```

### **9.2. Containerization**

Create a `Dockerfile` for containerization.

**Example `Dockerfile`:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "main:app", "-b", "0.0.0.0:8000"]
```

### **9.3. Security**

- Ensure all secrets are managed securely.
- Use HTTPS for all communications.

---

## **Conclusion**

By following this comprehensive guide, you've built a production-ready Bonzo conversational bot that:

- **Handles complex conversation payloads** using Pydantic models.
- **Interacts with Bonzo's API** to send messages and add notes.
- **Uses OpenAI's LLMs** to craft responses and ensure compliance.
- **Includes unit tests** for critical components.
- **Tailors prompts** for the specific use case of sales qualification.

---

## **Appendix: Sample Conversation Payloads**

**Example Conversation Payload:**

```json
{
  "id": 258673865,
  "prospect_id": 36194136,
  "created_at": "2022-01-01T01:00:00.000000Z",
  "content": "test error send",
  "type": "sms",
  "direction": "outgoing",
  "status": "not_delivered",
  "user_name": "Ian Melchor",
  "user_data_extension": {
    "user_id": "123",
    "avatar_url": "https://example.com/avatar.jpg"
  },
  "metadata_extension": {
    "error_message": "Prospect phone number doesn't support SMS messages"
  }
}
```

---

**Note:** Remember to handle exceptions and errors appropriately in production code. Implement proper logging and monitoring to keep track of the application's performance and issues.

---

# **Final Checklist for Your Pair Programmer**

- **Project Setup:**
  - [ ] Create the project directory and initialize a virtual environment.
  - [ ] Install required packages from `requirements.txt`.
  - [ ] Set up environment variables in `.env`.

- **Data Models:**
  - [ ] Define `BonzoConversationModel` and related extensions in `models/conversation.py` using Pydantic.

- **API Integrations:**
  - [ ] Implement Bonzo API functions in `bonzo_api.py`, including conversation history handling with Pydantic models.
  - [ ] Implement OpenAI API setup in `openai_api.py`.

- **Agents Development:**
  - [ ] Develop orchestrator agent considering conversation payloads and use case-specific prompts.
  - [ ] Develop conversation agent that uses conversation history and tailored prompts for sales qualification.
  - [ ] Develop policy compliance agent with structured outputs and specific compliance policies.
  - [ ] Develop knowledge agent with simulated knowledge retrieval or integrate with actual knowledge base.

- **Main Application Logic:**
  - [ ] Implement the Flask app in `main.py`, handling incoming messages and coordinating agents.
  - [ ] Ensure conversation history is correctly passed to agents.

- **Unit Tests:**
  - [ ] Write unit tests for Bonzo API functions in `tests/test_bonzo_api.py`.
  - [ ] Write unit tests for agents in `tests/test_agents.py`.
  - [ ] Write unit tests for Pydantic models in `tests/test_models.py`.
  - [ ] Run tests and ensure all pass.

- **Testing and Deployment:**
  - [ ] Run the application locally and test with simulated incoming messages.
  - [ ] Consider deployment options and security best practices.

- **Additional Considerations:**
  - [ ] Implement proper error handling and logging.
  - [ ] Extend knowledge agent with actual retrieval logic as needed.
  - [ ] Monitor and log performance and issues in production.

---

Feel free to copy and paste the code examples into your project files. Adjust as necessary to fit your specific requirements. Let me know if you need further assistance!