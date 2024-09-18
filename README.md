# Bonzo Conversational Bot

This project implements a production-ready conversational chatbot for Bonzo CRM, designed to handle sales qualification needs using Pydantic models and OpenAI's Language Models.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Testing](#testing)
8. [Deployment](#deployment)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

The Bonzo Conversational Bot:

- Receives incoming messages from prospects, including detailed conversation payloads.
- Parses and understands the conversation payloads using Pydantic models.
- Determines the appropriate action using an orchestrator agent.
- Crafts responses using OpenAI's Language Models (LLMs) tailored for sales qualification needs.
- Sends SMS messages and adds notes to prospects via Bonzo's API.
- Ensures compliance with company policies.
- Logs interactions for future analysis.

## Prerequisites

- Python 3.8 or higher
- Bonzo API account and API key
- OpenAI API account and API key
- Basic understanding of Python, RESTful APIs, and data modeling

## Project Structure
