[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=18726060)

# London Population Explorer - COMP0035 Coursework 02

# NOTE: Github CI failure due to recent github changes to payment plans

A Flask web application for exploring and analyzing London population and immigration data.

## Project Structure

```text
project/
├── .gitignore             # Git ignore file
├── README.md              # Project description and instructions
├── requirements.txt       # List of dependencies
├── pyproject.toml         # Installation and package details
├── src/                   # Main code directory
│   ├── population_app/    # App package directory
│      ├── __init__.py     # Flask app configuration 
│      ├── static/         # CSS, JS and static assets
│      ├── templates/      # Jinja page templates (.html files)
│      ├── models.py       # Database models
│      ├── routes.py       # Main application routes
│      ├── api_routes.py   # API endpoints
│      ├── forms.py        # Form definitions
│      ├── helpers.py      # Helper functions
│      ├── charts.py       # Chart generation logic
│      ├── ml.py           # Machine learning predictions
│      ├── chatbot.py      # Chatbot functionality
│      ├── run.py          # Alternative for running Flask app
├── tests/                 # Test suite
│   ├── conftest.py        # Test fixtures
│   ├── test_api_routes.py # API route tests
│   ├── test_forms.py      # Form validation tests
│   ├── test_models.py     # Database model tests
│   ├── test_routes.py     # Route tests
│   ├── test_selenium.py   # Integration tests with Selenium
```

## Setup Instructions

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the project:
   ```
   pip install -e .
   ```
   ```
   pip install -r requirements.txt
   ```
4. Run the Flask app:
   ```
   flask --app population_app
   ```
   Visit http://127.0.0.1:5000/ in a browser to use the application.

   Alternatively, run using:
   ```
   python src/population_app/run.py
   ```

5. Run tests:
   ```
   pytest
   ```
Or to generate a coverage report:
   ```
   pytest --cov
   ```


## Features

- Interactive dashboard with population visualizations
- Data filtering and export functionality
- Population trends and prediction analysis
- REST API for programmatic data access
- Chatbot interface for natural language queries
- Admin interface for data management
