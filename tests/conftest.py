"""Pytest configuration and fixtures for population analysis application."""

import os
import tempfile
import pytest
from population_app import create_app
from population_app.models import db as _db
from datetime import datetime


@pytest.fixture(scope="session")
def app():
    """Create and configure a Flask app for testing.

    Returns:
        Flask: A configured Flask application for testing.
    """
    # Create a temporary file to use as a test database
    db_fd, db_path = tempfile.mkstemp()

    try:
        # Create the Flask app with test configuration
        app = create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
                "WTF_CSRF_ENABLED": False,  # Disable CSRF for testing
                "DATABASE_ERROR": None,  # Make sure there's no database error
            }
        )

        # Create an application context for the tests
        with app.app_context():
            # Create the database tables
            _db.create_all()

            # Initialize the database with minimal test data
            try:
                _init_test_data()
            except Exception:
                # Silently handle initialization errors
                pass

        yield app

        # Clean up database connections
        with app.app_context():
            if hasattr(_db.session, "close"):
                _db.session.close()
            _db.engine.dispose()

    finally:
        # Close file descriptor
        os.close(db_fd)
        # Try to remove file, but don't fail if can't
        try:
            os.unlink(db_path)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not delete temporary test database: {e}")


@pytest.fixture(scope="function")
def client(app):
    """Create a test client for the app.

    Args:
        app (Flask): The Flask application fixture.

    Returns:
        FlaskClient: A test client for making requests.
    """
    return app.test_client()


@pytest.fixture(scope="function")
def runner(app):
    """Create a test CLI runner for the app.

    Args:
        app (Flask): The Flask application fixture.

    Returns:
        FlaskCliRunner: A test CLI runner.
    """
    return app.test_cli_runner()


@pytest.fixture(scope="function")
def db_session(app):
    """Create a fresh database session for each test.

    Args:
        app (Flask): The Flask application fixture.

    Returns:
        Session: A database session for testing.
    """
    with app.app_context():
        # Store the original session
        original_session = _db.session

        # Create a new connection and begin a transaction
        connection = _db.engine.connect()
        transaction = connection.begin()

        # Create a new session
        from sqlalchemy.orm import Session

        session = Session(bind=connection)

        # Replace the original session
        _db.session = session

        yield session

        # Clean up properly
        session.close()
        transaction.rollback()
        connection.close()

        # Restore the original session
        _db.session = original_session


def _init_test_data():
    """Initialize database with minimal test data."""
    from population_app.models import (
        Location,
        Year,
        Demographic,
        Gender,
        MetricType,
        PopulationMetric,
        ChatbotQuery,
    )

    # Create test locations
    locations = [
        Location(location_id=1, area_code="LON", area_name="London"),
        Location(location_id=2, area_code="CAM", area_name="Camden"),
        Location(location_id=3, area_code="HAR", area_name="Harrow"),
    ]
    _db.session.add_all(locations)

    # Create test years
    years = [
        Year(year_id=1, year=2020),
        Year(year_id=2, year=2021),
        Year(year_id=3, year=2022),
    ]
    _db.session.add_all(years)

    # Create test demographics
    demographics = [
        Demographic(demographic_id=1, age=0),
        Demographic(demographic_id=2, age=25),
        Demographic(demographic_id=3, age=65),
    ]
    _db.session.add_all(demographics)

    # Create test genders
    genders = [
        Gender(gender_id=1, gender_type="male"),
        Gender(gender_id=2, gender_type="female"),
        Gender(gender_id=3, gender_type="all"),
    ]
    _db.session.add_all(genders)

    # Create test metric types
    metrics = [
        MetricType(metric_id=1, metric_name="population", category="base"),
        MetricType(metric_id=2, metric_name="births", category="vital"),
        MetricType(metric_id=3, metric_name="deaths", category="vital"),
        MetricType(metric_id=4, metric_name="international_net", category="migration"),
    ]
    _db.session.add_all(metrics)

    # Create test population metrics
    population_metrics = [
        # London population (all genders, all ages)
        PopulationMetric(
            metric_data_id=1,
            location_id=1,
            demographic_id=2,
            gender_id=3,
            year_id=3,
            metric_id=1,
            value=8900000,
        ),
        # Camden population
        PopulationMetric(
            metric_data_id=2,
            location_id=2,
            demographic_id=2,
            gender_id=3,
            year_id=3,
            metric_id=1,
            value=270000,
        ),
        # Harrow population
        PopulationMetric(
            metric_data_id=3,
            location_id=3,
            demographic_id=2,
            gender_id=3,
            year_id=3,
            metric_id=1,
            value=250000,
        ),
        # Add birth data
        PopulationMetric(
            metric_data_id=4,
            location_id=1,
            demographic_id=1,
            gender_id=3,
            year_id=3,
            metric_id=2,
            value=105000,
        ),
        # Add migration data
        PopulationMetric(
            metric_data_id=5,
            location_id=1,
            demographic_id=2,
            gender_id=3,
            year_id=3,
            metric_id=4,
            value=45000,
        ),
    ]
    _db.session.add_all(population_metrics)

    # Add test chatbot query
    test_query = ChatbotQuery(
        query_id=1,
        user_query="What is the population of London?",
        response=(
            "Based on the data, the population of London " "is 8,900,000 as of 2022."
        ),
        timestamp=datetime(2023, 1, 1, 12, 0, 0),
    )
    _db.session.add(test_query)

    _db.session.commit()
