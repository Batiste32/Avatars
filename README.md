# Interactive Avatar Overlay

A python program creating a PyQT powered interface allowing you to interact with custom AI powered characters.

## Setup

### Python Installation

This program was made using Python 3.10.6 and CUDA 12.6 for Windows.

Creating a venv is recommended :

```bash
python venv .venv
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
A command panel will open up asking you to input the name of the character to load.
