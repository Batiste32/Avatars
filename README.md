# Interactive Avatar Overlay

A python program creating a PyQT powered interface allowing you to interact with custom AI powered characters.

## Setup

### Install Ollama

This program uses Ollama as the LLM server. You can find it and download it here : [Ollama](https://ollama.com/download)
Once it is installed, you'll want to download the llava model.
Simply run in a command prompt `ollama pull llava`.
If the installation is successful, you can close the prompt.

### Python Installation

This program was made using Python 3.10.6 and CUDA 12.6 for Windows.

You can execute the `setup.bat` file to automatically install the environment.

If you want to manually install the environment, follow these steps in a command prompt :

Creating a venv is recommended :

```bash
python -m venv .venv
```

Activate your venv :

```bash
call .venv/Scripts/activate
```

Then install the requirements

```python
pip install -r requirements.txt
```

### Creating a character

A character is defined by two files : a text file containing the description of your character and an optional thumbnail image, grouped into a single folder with the same name as your character.

Example :

```file
avatars/
└── characters/
    ├── character_1/
    │   ├── context.txt
    │   └── character.png
    ├── character_2/
    │   ├── context.txt
    │   └── character.png
    └── ...
```

## Usage

You can open the `launcher.bat` file to directly open the program, given you've followed the installation step.
A command panel will open up asking you to input the name of the character to load. If you don't specify the name and skip this question, the last character loaded will be used.
