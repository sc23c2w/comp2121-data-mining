#!/bin/bash
cd docker/llm-api
docker build -t llm-api .
docker run -d -p 8000:8000 --name llm-api-container llm-api
echo "LLM API running at http://localhost:8000"