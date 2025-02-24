#!/bin/bash

# Create main project directories
mkdir -p app/utils app/components tests deployment

# Create necessary files
touch app/main.py
touch app/utils/__init__.py
touch app/utils/gcs.py
touch app/utils/parser.py
touch app/utils/embeddings.py
touch app/utils/chatbot.py
touch app/components/__init__.py
touch tests/__init__.py
touch deployment/Dockerfile
touch deployment/app.yaml