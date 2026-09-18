# Text Summarizer Application

A full-stack web application that leverages natural language processing to condense long conversations, chat logs, and transcripts into concise, actionable summaries. 

![Text Summarizer Interface](Screenshot%202026-09-18%20143512.png)

## 📌 Project Overview

This project is divided into two primary components:
1. **Machine Learning Model (`Text Summarizer/`):** Contains the Jupyter Notebook and training datasets used to fine-tune a summarization model. 
2. **Web Application (`TextSummarizerAPP/`):** A lightweight Flask-based backend serving a clean, responsive frontend interface. 

The application is specifically optimized for dialogue and chat transcript summarization, reading multi-turn conversations and outputting the core decisions and action items in a short paragraph format.

## 📁 Repository Structure

```text
Text-summarizer-project/
├── Text Summarizer/                 # Machine Learning & Model Training
│   ├── samsum-train.csv             # Training dataset
│   ├── samsum-test.csv              # Testing dataset
│   ├── samsum-validation.csv        # Validation dataset
│   └── text_summarizer.ipynb        # Jupyter Notebook with training pipeline
│
├── TextSummarizerAPP/               # Web Application
│   ├── app.py                       # Flask backend application
│   ├── index.html                   # Frontend user interface
│   └── saved_summary_model/         # Directory for the generated model files (ignored in Git)
│
└── .gitignore                       # Git ignore file for large datasets and environments
🛠️ Tech Stack
Backend: Python, Flask

Frontend: HTML5, CSS3, JavaScript

Machine Learning: Hugging Face Transformers, PyTorch

Data Processing: Pandas, Jupyter Notebook

🚀 Getting Started
Prerequisites
Python 3.8+

Git

Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/shreyaagarwal156/Text-summarizer-project.git](https://github.com/shreyaagarwal156/Text-summarizer-project.git)
cd Text-summarizer-project
Generate the Model (First-time setup):

Navigate to the Text Summarizer/ directory.

Open and run all cells in text_summarizer.ipynb to train the model on the provided SAMSum datasets.

Once training is complete, move the generated model files (e.g., model.safetensors, config.json, tokenizer.json) into the TextSummarizerAPP/saved_summary_model/ directory. (Note: These large files are intentionally excluded from the GitHub repository).

Set up the Web Application:

Navigate to the app directory:

Bash
cd TextSummarizerAPP
Create and activate a virtual environment:

Bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
Install the required dependencies (Flask, Transformers, PyTorch, etc.).

Run the Application:

Bash
python app.py
Open your browser and navigate to the local server address provided in the terminal (typically http://127.0.0.1:5000) to access the interface.

👨‍💻 Developer
Developed by Shreya Agarwal.