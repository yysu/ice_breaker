
# ice_breaker

A repository for learning LangChain🦜🔗  by building a generative ai application.

This is a web application crawling Linkedin data about a person and customizes an ice breaker with them.


![Logo](https://github.com/emarco177/ice_breaker/blob/main/static/demo.gif)
[![udemy](https://img.shields.io/badge/LangChain%20Udemy%20Course-Coupon%20%2412.99-brightgreen)](https://www.udemy.com/course/langchain/?referralCode=D981B8213164A3EA91AC)

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

``

`OPENAI_API_KEY`

`PROXYCURL_API_KEY`

`TAVILY_API_KEY`

`LANGCHAIN_TRACING_V2`  # Optional, for LangSmith tracing

`LANGCHAIN_API_KEY` # Optional

`LANGCHAIN_PROJECT` # Optional

To run this project, you will need to add the following environment variables to your .env file:

> **Important Note**: If you enable tracing by setting `LANGCHAIN_TRACING_V2=true`, you must have a valid LangSmith API key set in `LANGCHAIN_API_KEY`. Without a valid API key, the application will throw an error. If you don't need tracing, simply remove or comment out these environment variables.
## Run Locally

Clone the project

```bash
  git clone https://github.com/emarco177/ice_breaker.git
```

Go to the project directory

```bash
  cd ice_breaker
```

Install dependencies

```bash
  pipenv install
```

Start the flask server

```bash
  pipenv run app.py
```


## Running Tests

To run tests, run the following command

```bash
  pipenv run pytest .
```


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://www.udemy.com/course/langchain/?referralCode=D981B8213164A3EA91AC)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/eden-marco/)

