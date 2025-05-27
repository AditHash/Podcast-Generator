# Podcast Generator

A tool for generating podcasts automatically from a topic.

## Introduction

Podcast Generator is an automated tool that creates podcast episodes based on user-provided topics. It generates scripts, supports multiple speakers, and can be easily customized for different podcast formats.

## Features

- Automated podcast creation from topics
- Customizable script generation
- Support for multiple speakers
- Easy-to-use Streamlit web interface
- Modular and extensible codebase
- Python 3.10+ support

## Prerequisites

- Python 3.10 or higher

## Installation

```bash
git clone https://github.com/AditHash/podcast-generator.git
cd podcast-generator
python3 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

To launch the web interface:

```bash
streamlit run Demo/streamlit_app.py
```

Follow the on-screen instructions to input your topic and generate a podcast episode.

## Configuration

Edit the configuration file in the project root to customize podcast settings such as default speakers, script templates, and output options.

## Project Structure

```
podcast-generator/
├── Demo/
│   └── streamlit_app.py
├── requirements.txt
├── README.md
├── ... (other source files)
```

- `Demo/streamlit_app.py`: Main Streamlit web application.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License.

## Contact

For questions or suggestions, please contact [AditHash](https://github.com/AditHash).


