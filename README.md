# Distributed Audio Interview Processing

## Description
This project aims to facilitate the distributed processing of audio interviews by converting them into text format for further analysis and processing of responses. The system utilizes Dask for distributed computing, Pydub for audio processing, and Hugging Face Transformers for automatic speech recognition (ASR) and question answering (QA) tasks.

## Features
- **Audio Splitting**: Splits the input audio file into smaller parts to manage processing efficiently.
- **Automatic Speech Recognition (ASR)**: Converts audio segments into text using pre-trained models from Hugging Face Transformers.
- **Spell Correction**: Optionally corrects spelling errors in the transcribed text using a spell correction model.
- **Question Answering (QA)**: Analyzes the transcribed text by answering predefined questions using a QA model.
- **Distributed Computing**: Utilizes Dask for parallel and distributed computing, enabling efficient processing of large audio files.

## Installation
1. Clone the repository: `git clone https://github.com/username/repository.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage
1. Place the audio file (in MP3 format) named `interview2.mp3` in the project directory.
2. Run the Python script `process_interview.py` to initiate the processing.
3. The script will split the audio file, transcribe each segment, correct spelling errors (optional), and perform question answering analysis.
4. The final combined text with answers to predefined questions will be saved in the `output/combined_text.txt` file.

## Dependencies
- dask
- transformers
- pydub
