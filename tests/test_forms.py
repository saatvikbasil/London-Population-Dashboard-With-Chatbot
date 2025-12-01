"""Form validation tests for population analysis application."""

import pytest
from population_app.forms import (
    FilterForm, DataExportForm, PredictionForm, ChatbotForm
)


def test_filter_form_validation(app):
    """Test FilterForm validation with valid data.

    GIVEN a FilterForm
    WHEN valid data is submitted
    THEN check that validation passes
    """
    with app.test_request_context():
        # Create form with valid data
        form = FilterForm(
            location=1,
            demographic=2,
            gender=3,
            start_year=1,
            end_year=3,
            metric_type=1
        )
        
        # Set choices for SelectFields
        form.location.choices = [
            (0, 'All Locations'), 
            (1, 'London')
        ]
        form.demographic.choices = [
            (0, 'All Ages'), 
            (2, '25')
        ]
        form.gender.choices = [
            (1, 'male'), 
            (2, 'female'), 
            (3, 'all')
        ]
        form.start_year.choices = [
            (0, 'All Years'), 
            (1, '2020')
        ]
        form.end_year.choices = [
            (0, 'All Years'), 
            (3, '2022')
        ]
        form.metric_type.choices = [
            (0, 'All Metrics'), 
            (1, 'Population')
        ]
        
        # All fields are optional in FilterForm
        assert form.validate() is True


def test_filter_form_empty_validation(app):
    """Test FilterForm validation with empty data.

    GIVEN a FilterForm
    WHEN empty data is submitted
    THEN check that validation still passes (all fields optional)
    """
    with app.test_request_context():
        # Create form with no data
        form = FilterForm()
        
        # Initialize choices for all SelectFields
        form.location.choices = [(0, 'All Locations')]
        form.demographic.choices = [(0, 'All Age Groups')]
        form.gender.choices = [(0, 'All Genders')]
        form.start_year.choices = [(0, 'All Years')]
        form.end_year.choices = [(0, 'All Years')]
        form.metric_type.choices = [(0, 'All Metrics')]
        
        # All fields are optional, so validation should pass
        assert form.validate() is True


def test_data_export_form_validation(app):
    """Test DataExportForm validation with valid formats.

    GIVEN a DataExportForm
    WHEN valid data is submitted
    THEN check that validation passes
    """
    with app.test_request_context():
        # Test CSV export
        form = DataExportForm(export_format='csv')
        assert form.validate() is True
        
        # Test Excel export
        form = DataExportForm(export_format='excel')
        assert form.validate() is True
        
        # Test JSON export
        form = DataExportForm(export_format='json')
        assert form.validate() is True


def test_data_export_form_invalid_format(app):
    """Test DataExportForm validation with invalid format.

    GIVEN a DataExportForm
    WHEN invalid format is submitted
    THEN check that validation fails
    """
    with app.test_request_context():
        # Create form with invalid data
        form = DataExportForm(export_format='invalid-format')
        assert form.validate() is False


def test_chatbot_form_validation(app):
    """Test ChatbotForm validation with valid query.

    GIVEN a ChatbotForm
    WHEN valid query is submitted
    THEN check that validation passes
    """
    with app.test_request_context():
        # Create form with valid data
        form = ChatbotForm(query="What is the population of London?")
        assert form.validate() is True


def test_chatbot_form_empty_validation(app):
    """Test ChatbotForm validation with empty query.

    GIVEN a ChatbotForm
    WHEN empty query is submitted
    THEN check that validation fails
    """
    with app.test_request_context():
        # Create form with no data
        form = ChatbotForm(query="")
        assert form.validate() is False