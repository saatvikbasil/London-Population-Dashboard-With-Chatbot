"""
Test models for the population_app application.
"""

from datetime import datetime
from population_app.models import (
    Location,
    Year,
    Demographic,
    Gender,
    MetricType,
    PopulationMetric,
    ChatbotQuery,
)


def test_location_model(db_session):
    """
    GIVEN a Location model
    WHEN a new Location is created
    THEN check the attributes are defined correctly
    """
    # Create a test location
    location = Location(location_id=100, area_code="TST", area_name="Test Location")

    db_session.add(location)
    db_session.commit()

    # Query the location back
    saved_location = db_session.query(Location).filter_by(location_id=100).first()

    # Check attributes
    assert saved_location.area_code == "TST"
    assert saved_location.area_name == "Test Location"
    assert saved_location.location_name == "Test Location"  # Test the property


def test_demographic_model(db_session):
    """
    GIVEN a Demographic model
    WHEN a new Demographic is created
    THEN check the attributes and property are defined correctly
    """
    # Create a test demographic
    demographic = Demographic(demographic_id=100, age=45)

    db_session.add(demographic)
    db_session.commit()

    # Query it back
    saved_demographic = (
        db_session.query(Demographic).filter_by(demographic_id=100).first()
    )

    # Check attributes
    assert saved_demographic.age == 45
    assert saved_demographic.age_group == "45"  # Test the property


def test_gender_model(db_session):
    """
    GIVEN a Gender model
    WHEN a new Gender is created
    THEN check the attributes are defined correctly
    """
    # Create a test gender
    gender = Gender(gender_id=100, gender_type="non-binary")

    db_session.add(gender)
    db_session.commit()

    # Query it back
    saved_gender = db_session.query(Gender).filter_by(gender_id=100).first()

    # Check attributes
    assert saved_gender.gender_type == "non-binary"


def test_year_model(db_session):
    """
    GIVEN a Year model
    WHEN a new Year is created
    THEN check the attributes and property are defined correctly
    """
    # Create a test year
    year = Year(year_id=100, year=2025)

    db_session.add(year)
    db_session.commit()

    # Query it back
    saved_year = db_session.query(Year).filter_by(year_id=100).first()

    # Check attributes
    assert saved_year.year == 2025
    assert saved_year.year_value == 2025  # Test the property


def test_metric_type_model(db_session):
    """
    GIVEN a MetricType model
    WHEN a new MetricType is created
    THEN check the attributes are defined correctly
    """
    # Create a test metric type
    metric_type = MetricType(metric_id=100, metric_name="test_metric", category="test")

    db_session.add(metric_type)
    db_session.commit()

    # Query it back
    saved_metric = db_session.query(MetricType).filter_by(metric_id=100).first()

    # Check attributes
    assert saved_metric.metric_name == "test_metric"
    assert saved_metric.category == "test"


# In test_models.py
def test_population_metric_model_relationships(db_session):
    """
    GIVEN a PopulationMetric model with relationships
    WHEN a new PopulationMetric is created with related models
    THEN check the relationships are established correctly
    """
    # Create test related models if they don't exist
    location = db_session.query(Location).filter_by(location_id=1).first()
    if not location:
        location = Location(location_id=1, area_code="LON", area_name="London")
        db_session.add(location)
        db_session.commit()

    demographic = db_session.query(Demographic).filter_by(demographic_id=2).first()
    if not demographic:
        demographic = Demographic(demographic_id=2, age=25)
        db_session.add(demographic)
        db_session.commit()

    gender = db_session.query(Gender).filter_by(gender_id=3).first()
    if not gender:
        gender = Gender(gender_id=3, gender_type="all")
        db_session.add(gender)
        db_session.commit()

    year = db_session.query(Year).filter_by(year_id=3).first()
    if not year:
        year = Year(year_id=3, year=2022)
        db_session.add(year)
        db_session.commit()

    metric_type = db_session.query(MetricType).filter_by(metric_id=1).first()
    if not metric_type:
        metric_type = MetricType(metric_id=1, metric_name="population", category="base")
        db_session.add(metric_type)
        db_session.commit()

    # Create a test population metric
    metric = PopulationMetric(
        metric_data_id=100,
        location_id=location.location_id,
        demographic_id=demographic.demographic_id,
        gender_id=gender.gender_id,
        year_id=year.year_id,
        metric_id=metric_type.metric_id,
        value=9000000,
    )

    db_session.add(metric)
    db_session.commit()

    # Query it back
    saved_metric = (
        db_session.query(PopulationMetric).filter_by(metric_data_id=100).first()
    )

    # Check relationships
    assert saved_metric.location.location_id == location.location_id
    assert saved_metric.demographic.demographic_id == demographic.demographic_id
    assert saved_metric.gender.gender_id == gender.gender_id
    assert saved_metric.year.year_id == year.year_id
    assert saved_metric.metric_type.metric_id == metric_type.metric_id
    assert saved_metric.value == 9000000


def test_chatbot_query_model(db_session):
    """
    GIVEN a ChatbotQuery model
    WHEN a new ChatbotQuery is created
    THEN check the attributes are defined correctly
    """
    # Create a test chatbot query
    timestamp = datetime.now()
    query = ChatbotQuery(
        query_id=100,
        user_query="How has London's population changed?",
        response="London's population has grown steadily since 2020.",
        timestamp=timestamp,
    )

    db_session.add(query)
    db_session.commit()

    # Query it back
    saved_query = db_session.query(ChatbotQuery).filter_by(query_id=100).first()

    # Check attributes
    assert saved_query.user_query == "How has London's population changed?"
    assert saved_query.response == "London's population has grown steadily since 2020."
    assert saved_query.timestamp == timestamp
