"""Population prediction module using machine learning techniques.

This module provides tools for training and generating population trend predictions
using polynomial regression and visualizing the results.
"""

import json
import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sqlalchemy import func
from population_app.models import (
    db,
    PopulationMetric,
    Location,
    Year,
    Demographic,
    Gender,
    MetricType,
    Prediction,
    PredictionModel,
)


class PopulationPredictor:
    """Machine learning model to predict future population trends.

    Uses polynomial regression to capture non-linear trends in population growth.

    Attributes:
        model: Polynomial regression pipeline
        degree: Degree of the polynomial model
        is_trained: Boolean indicating if the model is trained
        train_score: R² score of the model on training data
    """

    def __init__(self, degree: int = 2):
        """Initialize the population predictor with a polynomial model.

        Args:
            degree (int, optional): Degree of the polynomial model. Defaults to 2.
        """
        self.model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        self.degree = degree
        self.is_trained = False
        self.train_score = None
        self.mse = None

    def train(self, years: List[int], populations: List[float]) -> float:
        """Train the model on historical population data.

        Args:
            years: List of years as integers (e.g., [2018, 2019, 2020])
            populations: Corresponding population values

        Returns:
            R² score of the model on training data

        Raises:
            ValueError: If fewer than 3 data points are provided
        """
        if len(years) < 3:
            raise ValueError("Need at least 3 data points to train the model")

        X = np.array(years).reshape(-1, 1)
        y = np.array(populations)

        self.model.fit(X, y)
        self.is_trained = True

        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.train_score = r2_score(y, y_pred)
        self.mse = mean_squared_error(y, y_pred)

        return self.train_score

    def predict(self, future_years: List[int]) -> List[float]:
        """Predict population for future years.

        Args:
            future_years: List of years to predict for

        Returns:
            Predicted population values

        Raises:
            RuntimeError: If the model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        X_future = np.array(future_years).reshape(-1, 1)
        predictions = self.model.predict(X_future)

        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)

        return predictions.tolist()

    def get_confidence(self) -> str:
        """Get a measure of model confidence based on training score.

        Returns:
            Confidence level descriptor
        """
        if not self.is_trained:
            return "Not trained"

        confidence_levels = [
            (0.95, "Very High"),
            (0.85, "High"),
            (0.70, "Moderate"),
            (0.50, "Low"),
        ]

        for threshold, level in confidence_levels:
            if self.train_score > threshold:
                return level

        return "Very Low"


def get_age_group_name(age: Optional[int]) -> str:
    """Convert age to decade group name.

    Args:
        age: Age value or None

    Returns:
        Formatted age group name
    """
    if age is None:
        return "All Ages"
    if age == 0:
        return "Births"

    start = (age // 10) * 10
    end = start + 9
    return f"{start}-{end}"


def train_prediction_model(
    location_id: int, demographic_id: int, gender_id: int, metric_id: int
) -> Tuple[
    Optional[PopulationPredictor], Optional[pd.DataFrame], Optional[PredictionModel]
]:
    """Train a prediction model for population trends.

    Args:
        location_id: ID of the location
        demographic_id: ID of the demographic group
        gender_id: ID of the gender
        metric_id: ID of the metric type

    Returns:
        Tuple containing the trained predictor, historical data DataFrame,
        and prediction model entry
    """
    try:
        # Check if this is a birth-related metric
        metric_type = MetricType.query.get(metric_id)
        is_birth_metric = bool(
            metric_type and "birth" in metric_type.metric_name.lower()
        )

        if is_birth_metric:
            # Find the demographic_id for age 0 (births)
            birth_demographic = (
                db.session.query(Demographic).filter(Demographic.age == 0).first()
            )

            if birth_demographic:
                # Override the demographic_id to use age 0
                demographic_id = birth_demographic.demographic_id

        # Construct base query for historical data
        base_query = db.session.query(
            Year.year, func.sum(PopulationMetric.value).label("value")
        ).join(PopulationMetric, Year.year_id == PopulationMetric.year_id)

        # Apply filters
        if location_id > 0:
            base_query = base_query.filter(PopulationMetric.location_id == location_id)

        if demographic_id > 0:
            base_query = base_query.filter(
                PopulationMetric.demographic_id == demographic_id
            )

        base_query = base_query.filter(
            PopulationMetric.gender_id == gender_id,
            PopulationMetric.metric_id == metric_id,
        )

        # Group and order
        base_query = base_query.group_by(Year.year).order_by(Year.year)

        # Execute the query
        results = base_query.all()

        # Convert to DataFrame for easier data manipulation
        df = pd.DataFrame(results, columns=["year", "value"])

        # Validate data sufficiency
        if len(df) < 3:
            return None, None, None

        # Create and train the predictor
        predictor = PopulationPredictor(degree=2)
        predictor.train(df["year"].tolist(), df["value"].tolist())

        # Extract model parameters
        polynomial_features = predictor.model.steps[0][1]
        linear_model = predictor.model.steps[1][1]

        parameters = {
            "coefficients": linear_model.coef_.tolist(),
            "intercept": float(linear_model.intercept_),
            "degree": predictor.degree,
            "r2_score": float(predictor.train_score),
            "mse": float(predictor.mse),
        }

        # Create unique model name
        model_name = f"Model-{location_id}-{demographic_id}-{gender_id}-{metric_id}"

        # Check if model exists and update it, or create new
        model_entry = PredictionModel.query.filter_by(model_name=model_name).first()

        if model_entry:
            # Update existing
            model_entry.parameters = json.dumps(parameters)
            model_entry.accuracy = predictor.train_score
            model_entry.created_at = datetime.datetime.utcnow()
        else:
            # Create new
            model_entry = PredictionModel(
                model_name=model_name,
                description=f"Polynomial regression model degree {predictor.degree}",
                parameters=json.dumps(parameters),
                accuracy=predictor.train_score,
                created_at=datetime.datetime.utcnow(),
            )
            db.session.add(model_entry)

        # Commit changes
        db.session.commit()

        return predictor, df, model_entry

    except Exception:
        return None, None, None


def make_prediction(
    location_id: int,
    demographic_id: int,
    gender_id: int,
    metric_id: int,
    years_ahead: int = 5,
) -> Tuple[Optional[str], str]:
    """Make population predictions for specified parameters.

    Args:
        location_id: ID of the location
        demographic_id: ID of the demographic group
        gender_id: ID of the gender
        metric_id: ID of the metric type
        years_ahead: Number of years to predict ahead

    Returns:
        Tuple of prediction figure JSON and confidence level
    """
    try:
        # Determine if this is a birth-related metric
        metric_type = MetricType.query.get(metric_id)
        is_birth_metric = bool(
            metric_type and "birth" in metric_type.metric_name.lower()
        )

        # Handle birth-related metrics
        if is_birth_metric:
            birth_demographic = (
                db.session.query(Demographic).filter(Demographic.age == 0).first()
            )

            if birth_demographic:
                demographic_id = birth_demographic.demographic_id

        # Retrieve location, gender, and metric details
        location = Location.query.get(location_id)
        gender = Gender.query.get(gender_id)

        # Handle demographic name
        if demographic_id == 0:
            demographic_name = "All Ages"
        else:
            demographic = Demographic.query.get(demographic_id)
            if demographic:
                age = getattr(demographic, "age", None)
                demographic_name = (
                    get_age_group_name(age)
                    if age is not None
                    else getattr(
                        demographic, "age_group", f"Demographic {demographic_id}"
                    )
                )
            else:
                demographic_name = f"Demographic {demographic_id}"

        if is_birth_metric:
            demographic_name = "Births"

        # Validate required objects
        if not all([location, gender, metric_type]) or (
            demographic_id != 0 and not demographic
        ):
            missing = [
                obj
                for obj, name in [
                    (location, "location"),
                    (demographic_id == 0 or demographic, "demographic"),
                    (gender, "gender"),
                    (metric_type, "metric_type"),
                ]
                if not obj
            ]
            return None, f"Invalid parameters: {', '.join(missing)} not found"

        # Train model
        predictor, historical_df, model_entry = train_prediction_model(
            location_id, demographic_id, gender_id, metric_id
        )

        # Validate model and data
        if (
            not predictor
            or not predictor.is_trained
            or historical_df is None
            or len(historical_df) == 0
        ):
            return None, "Insufficient data or error training prediction model"

        # Get latest year from historical data
        latest_year = historical_df["year"].max()

        if latest_year is None:
            return None, "No year data found"

        # Generate future years
        future_years = list(range(latest_year + 1, latest_year + years_ahead + 1))

        # Make predictions
        try:
            predictions = predictor.predict(future_years)
        except Exception as e:
            return None, f"Error making predictions: {str(e)}"

        # Create prediction DataFrame
        prediction_df = pd.DataFrame(
            {"year": future_years, "predicted_value": predictions}
        )

        # Store predictions in database
        if model_entry:
            try:
                # Remove existing predictions for these years
                Prediction.query.filter(
                    Prediction.model_id == model_entry.model_id,
                    Prediction.year_value.in_(future_years),
                ).delete(synchronize_session=False)

                # Add new predictions
                for _, row in prediction_df.iterrows():
                    prediction = Prediction(
                        model_id=model_entry.model_id,
                        location_id=location_id,
                        demographic_id=demographic_id if demographic_id != 0 else None,
                        gender_id=gender_id,
                        year_value=int(row["year"]),
                        metric_id=metric_id,
                        predicted_value=float(row["predicted_value"]),
                        confidence_interval=float(row["predicted_value"] * 0.1),
                    )
                    db.session.add(prediction)

                db.session.commit()
            except Exception:
                db.session.rollback()

        # Create prediction visualization
        fig = go.Figure()

        # Add historical data
        fig.add_trace(
            go.Scatter(
                x=historical_df["year"].tolist(),
                y=historical_df["value"].tolist(),
                mode="lines+markers",
                name="Historical Data",
                line=dict(color="#1976D2", width=3),
            )
        )

        # Add prediction
        fig.add_trace(
            go.Scatter(
                x=prediction_df["year"].tolist(),
                y=prediction_df["predicted_value"].tolist(),
                mode="lines+markers",
                name="Prediction",
                line=dict(color="#e53935", width=3, dash="dash"),
            )
        )

        # Add confidence interval
        upper_bound = [
            value + value * 0.1 for value in prediction_df["predicted_value"].tolist()
        ]
        lower_bound = [
            max(0, value - value * 0.1)
            for value in prediction_df["predicted_value"].tolist()
        ]

        fig.add_trace(
            go.Scatter(
                x=prediction_df["year"].tolist(),
                y=upper_bound,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prediction_df["year"].tolist(),
                y=lower_bound,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(229, 57, 53, 0.2)",
                name="Confidence Interval",
            )
        )

        # Get display names
        location_name = getattr(
            location,
            "location_name",
            getattr(location, "area_name", f"Location {location_id}"),
        )
        gender_name = getattr(gender, "gender_type", f"Gender {gender_id}")
        metric_name = getattr(metric_type, "metric_name", f"Metric {metric_id}")

        # Add layout details
        fig.update_layout(
            title=f"Population Prediction for {location_name}, {demographic_name}, {gender_name}",
            xaxis_title="Year",
            yaxis_title=metric_name,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        # Convert to JSON safely
        try:
            fig_json = fig.to_json()
        except Exception:
            # Fallback to simpler JSON
            fig_data = {
                "data": [
                    {
                        "x": historical_df["year"].tolist(),
                        "y": historical_df["value"].tolist(),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Historical Data",
                    },
                    {
                        "x": prediction_df["year"].tolist(),
                        "y": prediction_df["predicted_value"].tolist(),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Prediction",
                    },
                ],
                "layout": {
                    "title": f"Population Prediction for {location_name}, {demographic_name}",
                    "xaxis": {"title": "Year"},
                    "yaxis": {"title": metric_name},
                },
            }
            fig_json = json.dumps(fig_data)

        # Return the figure as JSON and confidence level
        return fig_json, predictor.get_confidence()

    except Exception:
        return None, "Error generating prediction"
