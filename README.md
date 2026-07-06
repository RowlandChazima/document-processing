# Document Processing with LangChain & Groq

## Project Overview

This project demonstrates how to process an unstructured text document using **LangChain** and **Groq**. The application reads a document about the Eiffel Tower, splits it into smaller chunks using a fixed-size text splitting strategy, extracts important information, and generates a concise summary.

The project uses LangChain to build a simple pipeline.

---

## Features

- Read an unstructured text document
- Apply a fixed-size text splitting strategy
- Extract key facts, including:
  - Dates
  - Places
  - People
  - Numbers
  - Events

- Generate a short summary of the document
- Return the extracted information in structured JSON format

---

## Tools Used

- Python 3.x
- LangChain
- LangChain Groq
- LangChain Text Splitters
- Pydantic
- Python Dotenv

---

## Project Structure

```text
document_processing/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd document_processing
```

### 2. Create a virtual environment

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configure the API Key

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Run the Project

```bash
python app.py
```

---

## Example Output

The application displays:

- The generated document chunks
- Extracted key facts (dates, places, people, numbers, and events)
- A concise summary of the document

---

## LangChain Workflow

The application follows these steps:

1. Load the document.
2. Split the document into smaller chunks using `RecursiveCharacterTextSplitter`.
3. Create a prompt instructing the LLM to extract important information.
4. Send the prompt to a Groq model through LangChain.
5. Parse the response into structured JSON using `JsonOutputParser`.
6. Display the extracted information and summary.

---

## Author

Rowland Chazima
