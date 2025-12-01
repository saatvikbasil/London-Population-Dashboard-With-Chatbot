"""
Models for the Population App using SQLAlchemy ORM.
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, DateTime
import datetime

db = SQLAlchemy()


class Location(db.Model):
    """Represents London borough/area locations"""

    __tablename__ = "Location"

    location_id = Column(Integer, primary_key=True)
    area_code = Column(String(10), nullable=False)
    area_name = Column(String(50), nullable=False)

    # Define property to maintain compatibility with existing code
    @property
    def location_name(self):
        return self.area_name

    # Relationships
    population_metrics = relationship("PopulationMetric", back_populates="location")

    def __repr__(self):
        return f"<Location {self.area_name}>"


class Demographic(db.Model):
    """Represents demographic age groups"""

    __tablename__ = "DemographicBase"

    demographic_id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)

    # Define property to maintain compatibility with existing code
    @property
    def age_group(self):
        return str(self.age)

    # Relationships
    population_metrics = relationship("PopulationMetric", back_populates="demographic")

    def __repr__(self):
        return f"<Demographic {self.age}>"


class Gender(db.Model):
    """Represents gender categories"""

    __tablename__ = "Gender"

    gender_id = db.Column(db.Integer, primary_key=True)
    gender_type = db.Column(db.String(10), nullable=False)

    # Relationships
    population_metrics = relationship("PopulationMetric", back_populates="gender")

    def __repr__(self):
        return f"<Gender {self.gender_type}>"


class Year(db.Model):
    """Represents years for time series data"""

    __tablename__ = "Year"

    year_id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)

    # Define property to maintain compatibility with existing code
    @property
    def year_value(self):
        return self.year

    # Relationships
    population_metrics = relationship("PopulationMetric", back_populates="year")

    def __repr__(self):
        return f"<Year {self.year}>"


class MetricType(db.Model):
    """Represents different types of population metrics"""

    __tablename__ = "MetricType"

    metric_id = db.Column(db.Integer, primary_key=True)
    metric_name = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50), nullable=False)

    # Relationships
    population_metrics = relationship("PopulationMetric", back_populates="metric_type")

    def __repr__(self):
        return f"<MetricType {self.metric_name}>"


class PopulationMetric(db.Model):
    """Core table storing population metric values"""

    __tablename__ = "PopulationMetrics"

    metric_data_id = db.Column(Integer, primary_key=True)
    location_id = db.Column(db.Integer, ForeignKey("Location.location_id"))
    demographic_id = db.Column(db.Integer, ForeignKey("DemographicBase.demographic_id"))
    gender_id = db.Column(db.Integer, ForeignKey("Gender.gender_id"))
    year_id = db.Column(db.Integer, ForeignKey("Year.year_id"))
    metric_id = db.Column(db.Integer, ForeignKey("MetricType.metric_id"))
    value = db.Column(db.Integer)

    # Relationships
    location = relationship("Location", back_populates="population_metrics")
    demographic = relationship("Demographic", back_populates="population_metrics")
    gender = relationship("Gender", back_populates="population_metrics")
    year = relationship("Year", back_populates="population_metrics")
    metric_type = relationship("MetricType", back_populates="population_metrics")

    def __repr__(self):
        return f"<PopulationMetric {self.metric_data_id}>"


# These models may not be used yet, but keeping them compatible with your original structure
class PredictionModel(db.Model):
    """Stores prediction model metadata"""

    __tablename__ = "prediction_model"

    model_id = Column(Integer, primary_key=True)
    model_name = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    parameters = Column(Text)  # Stored as JSON
    accuracy = Column(Float)

    # Relationships
    predictions = relationship("Prediction", back_populates="model")

    def __repr__(self):
        return f"<PredictionModel {self.model_name}>"


class Prediction(db.Model):
    """Stores prediction results"""

    __tablename__ = "prediction"

    prediction_id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("prediction_model.model_id"))
    location_id = Column(Integer, ForeignKey("Location.location_id"))
    demographic_id = Column(Integer, ForeignKey("DemographicBase.demographic_id"))
    gender_id = Column(Integer, ForeignKey("Gender.gender_id"))
    year_value = Column(Integer, nullable=False)  # Future year
    metric_id = Column(Integer, ForeignKey("MetricType.metric_id"))
    predicted_value = Column(Float, nullable=False)
    confidence_interval = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Relationships
    model = relationship("PredictionModel", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction {self.prediction_id}>"


class ChatbotQuery(db.Model):
    """Stores user chatbot queries for analysis"""

    __tablename__ = "chatbot_query"

    query_id = Column(Integer, primary_key=True)
    user_query = Column(Text, nullable=False)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<ChatbotQuery {self.query_id}>"


class User(db.Model):
    """Admin users for the system"""

    __tablename__ = "user"

    user_id = Column(Integer, primary_key=True)
    username = Column(String(100), nullable=False, unique=True)
    email = Column(String(100), nullable=False, unique=True)
    role = Column(String(50), default="viewer")  # viewer, admin

    def __repr__(self):
        return f"<User {self.username}>"


class DataUpload(db.Model):
    """Tracks data uploads by admins"""

    __tablename__ = "data_upload"

    upload_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.user_id"))
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String(50))  # processing, completed, error
    error_message = Column(Text)

    def __repr__(self):
        return f"<DataUpload {self.upload_id}>"
