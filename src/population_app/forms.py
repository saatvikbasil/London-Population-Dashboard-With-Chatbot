"""Flask WTForms for population data application.

This module defines form classes for various interactions in the population
data management and analysis application.
"""

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SelectField, IntegerField, SubmitField
from wtforms.validators import DataRequired, Optional, NumberRange


class FilterForm(FlaskForm):
    """Form for filtering population data.

    Allows users to apply various filters to population metrics.

    Attributes:
        location (SelectField): Select a specific location.
        demographic (SelectField): Select an age group.
        gender (SelectField): Select a gender.
        start_year (SelectField): Select the start year for filtering.
        end_year (SelectField): Select the end year for filtering.
        metric_type (SelectField): Select a metric type.
        submit (SubmitField): Submit button to apply filters.
    """

    location = SelectField("Location", validators=[Optional()], coerce=int)
    demographic = SelectField("Age Group", validators=[Optional()], coerce=int)
    gender = SelectField("Gender", validators=[Optional()], coerce=int)
    start_year = SelectField("From Year", validators=[Optional()], coerce=int)
    end_year = SelectField("To Year", validators=[Optional()], coerce=int)
    metric_type = SelectField("Metric Type", validators=[Optional()], coerce=int)
    submit = SubmitField("Apply Filters")


class DataExportForm(FlaskForm):
    """Form for exporting filtered data.

    Allows users to choose an export format for population data.

    Attributes:
        export_format (SelectField): Select the desired export format.
        submit (SubmitField): Submit button to export data.
    """

    export_format = SelectField(
        "Export Format",
        choices=[("csv", "CSV"), ("excel", "Excel"), ("json", "JSON")],
        validators=[DataRequired()],
    )
    submit = SubmitField("Export Data")


class PredictionForm(FlaskForm):
    """Form for generating population predictions.

    Allows users to create population forecasts based on selected parameters.

    Attributes:
        location (SelectField): Select a location for prediction.
        demographic (SelectField): Select an age group for prediction.
        gender (SelectField): Select a gender for prediction.
        metric_type (SelectField): Select a metric type for prediction.
        years_ahead (IntegerField): Number of years to predict ahead.
        submit (SubmitField): Submit button to generate prediction.
    """

    location = SelectField("Location", validators=[DataRequired()], coerce=int)
    demographic = SelectField("Age Group", validators=[DataRequired()], coerce=int)
    gender = SelectField("Gender", validators=[DataRequired()], coerce=int)
    metric_type = SelectField("Metric Type", validators=[DataRequired()], coerce=int)
    years_ahead = IntegerField(
        "Predict Years Ahead",
        validators=[DataRequired(), NumberRange(min=1, max=20)],
        default=5,
    )
    submit = SubmitField("Generate Prediction")


class ChatbotForm(FlaskForm):
    """Form for chatbot queries.

    Allows users to submit queries to the population data chatbot.

    Attributes:
        query (StringField): Input field for user's query.
        submit (SubmitField): Submit button to send query.
    """

    query = StringField(
        "Ask about London population data:", validators=[DataRequired()]
    )
    submit = SubmitField("Ask")


class DataUploadForm(FlaskForm):
    """Form for admin data uploads.

    Allows administrators to upload population-related data files.

    Attributes:
        data_file (FileField): File upload field for data files.
        data_type (SelectField): Select the type of data being uploaded.
        submit (SubmitField): Submit button to upload data.
    """

    data_file = FileField(
        "Upload Data File",
        validators=[
            FileRequired(),
            FileAllowed(["csv", "xlsx"], "CSV or Excel files only!"),
        ],
    )
    data_type = SelectField(
        "Data Type",
        choices=[
            ("population", "Population Data"),
            ("location", "Location Data"),
            ("demographic", "Demographic Data"),
        ],
    )
    submit = SubmitField("Upload")
