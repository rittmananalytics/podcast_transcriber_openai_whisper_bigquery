# OpenAI Whisper API-Powered Podcast Transcription, Labelling and Remixing

This Jupyter notebook script automates the process of downloading, transcribing, and analyzing podcast episodes. 
It uses OpenAI's Whisper for transcription, GPT-4 for content analysis, and stores the results in Google BigQuery.

## Features

- Downloads podcast episodes from an RSS feed
- Transcribes audio using OpenAI's Whisper model
- Classifies episodes based on content
- Labels speakers in the transcript
- Generates a summary of insights, opinions, and trends discussed in the episode
- Extracts notable quotes from the guest speaker
- Stores all processed data in Google BigQuery

## Prerequisites

Before using this notebook, you need to have:

1. An OpenAI API key
2. A Google Cloud Platform account with BigQuery enabled
3. A service account key file for Google Cloud Platform

## Required Libraries

The script uses the following Python libraries:

- requests
- pandas
- feedparser
- pydub
- openai
- google-cloud-bigquery

You can install these libraries using pip:
pip install requests pandas feedparser pydub openai google-cloud-bigquery
Copy
Note: You also need to have FFmpeg installed on your system for audio processing.

## Configuration

Before running the notebook, you need to configure the following:

1. OpenAI API Key:
   Replace `'your-openai-api-key-here'` with your actual OpenAI API key.

2. Google Cloud Service Account Key:
   Replace `'path/to/your/service-account-key.json'` with the path to your service account key file.

3. BigQuery Dataset and Table:
   Set the `dataset_name` and `table_name` variables to your desired values.

4. Podcast Host Name:
   Set the `podcast_host_name` to the desired host of the podcast.

6. Podcast RSS Feed:
   Set the `feed_url` variable to the RSS feed URL of the podcast you want to process.

7. Number of Episodes:
   Set the `num_episodes` variable to the number of recent episodes you want to process.

## How It Works

1. The script fetches podcast episode information from the provided RSS feed.
2. For each episode:
   - It downloads the audio file
   - Transcribes the audio using OpenAI's Whisper
   - Classifies the episode content using GPT-4
   - Labels speakers in the transcript
   - Generates a summary and extracts insights using GPT-4
3. All processed data is then stored in the specified BigQuery table.

## Usage

1. Open the Jupyter notebook in your preferred environment (e.g., JupyterLab, Google Colab).
2. Configure the variables as described in the Configuration section.
3. Run all cells in the notebook.

The script will process the specified number of episodes and store the results in BigQuery. You can then query this data for further analysis or use in other applications.

## Output

The script creates a BigQuery table with the following columns:

- title
- link
- description
- published
- audio_url
- transcript
- classification
- summary_and_insights

## Limitations and Considerations

- Processing time can be significant, especially for longer episodes or when processing many episodes at once.
- Be mindful of API usage costs for both OpenAI and Google Cloud Platform.
- Ensure you have appropriate permissions and comply with the terms of service for all APIs and services used.
- The script currently does not handle pagination for RSS feeds with a large number of episodes.

# Podcast Transcriber

This Jupyter notebook generates simulated podcast conversations about data analytics, business intelligence, and modern data stack technologies. It uses OpenAI's GPT-4 model to create realistic discussions between a host and various guests, based on profiles generated from actual podcast transcripts.

## Features

- Retrieves podcast data from a BigQuery table
- Identifies guests from podcast episode titles
- Generates detailed guest profiles using GPT-4
- Creates diverse discussion topics related to data analytics
- Simulates conversations between the host and selected guests
- Stores guest profiles and simulated conversations in BigQuery tables
- Outputs conversations in both JSON and HTML formats

## Prerequisites

- Google Cloud Platform account with BigQuery enabled
- OpenAI API key
- Python 3.7+

## Required Libraries

- google-cloud-bigquery
- pandas
- openai
- json
- random

## Setup

1. Clone this repository to your local machine or Google Colab environment.
2. Install the required libraries:
pip install google-cloud-bigquery pandas openai
Copy3. Set up your Google Cloud credentials and OpenAI API key as environment variables or update the placeholders in the notebook.
4. Update the BigQuery project ID, dataset name, and table names in the notebook to match your GCP setup.

## Usage

1. Run the notebook cells in order.
2. The script will:
- Create necessary BigQuery tables if they don't exist
- Retrieve podcast data from the specified BigQuery table
- Identify guests and generate their profiles
- Generate 10 diverse discussion topics
- Simulate conversations for each topic
- Store the results in BigQuery tables

## BigQuery Table Schemas

### Guest Profiles Table

- `guest_name` (STRING): Name of the guest
- `profile` (STRING): JSON string containing the guest's profile

### Simulated Conversations Table

- `topic` (STRING): The discussion topic
- `guests` (STRING): JSON array of guest names
- `guest_profiles` (STRING): JSON object containing profiles of participating guests
- `conversation` (STRING): JSON array of the simulated conversation
- `html_conversation` (STRING): HTML-formatted version of the conversation

## Customization

- Adjust the number of simulated conversations by changing the range in the main execution loop.
- Modify the `generate_topic` function to focus on specific areas of interest.
- Tweak the `select_relevant_guests` function to change how guests are selected for each conversation.

## Notes

- The quality and diversity of the simulated conversations depend on the input data and the performance of the GPT-4 model.
- Be mindful of API usage costs when running this notebook, especially when generating a large number of conversations.
- Ensure compliance with OpenAI's use-case policies and your organization's data handling guidelines.

