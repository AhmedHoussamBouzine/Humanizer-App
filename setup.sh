#!/usr/bin/env bash

echo "Installing pip dependencies..."
pip install -r requirements.txt

echo "Installing spaCy model..."
python -m spacy download fr_core_news_sm
python -m nltk.downloader all

