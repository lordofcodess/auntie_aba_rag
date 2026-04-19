#!/bin/bash
source venv/bin/activate

# Check if API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set"
    echo "Get a free key: https://aistudio.google.com/app/apikey"
    echo "Then run: export GEMINI_API_KEY='your-key-here'"
    exit 1
fi

# Run test query
python rag_chat.py "What Computer Science courses are offered at Level 200?"
