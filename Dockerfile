FROM python:3.9

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install pinecone-client \
                langchain-huggingface \
                langchain-pinecone \
                langchain-openai \
                langgraph \
                transformers \
                torch \
                flask

COPY . /app

EXPOSE 5000

CMD ["python", "final.py"]
