"""Routes for population analysis web application."""

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    current_app,
    send_file,
)
from sqlalchemy import func
from population_app.models import (
    db,
    Location,
    Year,
    Demographic,
    Gender,
    MetricType,
    PopulationMetric,
    ChatbotQuery,
)
from population_app.forms import (
    FilterForm,
    DataExportForm,
    PredictionForm,
    ChatbotForm,
    DataUploadForm,
)
from population_app.helpers import get_filter_choices
from population_app.charts import create_dashboard_charts
import pandas as pd
import plotly
import json
from io import BytesIO

# Create blueprints
main_bp = Blueprint("main", __name__)
admin_bp = Blueprint("admin", __name__)


def handle_404(error):
    """Handle 404 Not Found errors.

    Args:
        error: The error object.

    Returns:
        tuple: Rendered error template and status code.
    """
    return (
        render_template("error.html", error_code=404, error_message="Page not found"),
        404,
    )


def handle_500(error):
    """Handle 500 Internal Server errors.

    Args:
        error: The error object.

    Returns:
        tuple: Rendered error template and status code.
    """
    return (
        render_template("error.html", error_code=500, error_message="Server error"),
        500,
    )


def apply_filters(query, filters):
    """Apply filters to a database query.

    Args:
        query (sqlalchemy.orm.query.Query): The base query to filter.
        filters (dict): Dictionary of filter parameters.

    Returns:
        sqlalchemy.orm.query.Query: Filtered query.
    """
    # Location filter
    if filters.get("location") and filters["location"] != 0:
        query = query.filter(PopulationMetric.location_id == filters["location"])

    # Demographic filter
    if filters.get("demographic") and filters["demographic"] != 0:
        query = query.filter(PopulationMetric.demographic_id == filters["demographic"])

    # Gender filter
    if filters.get("gender") and filters["gender"] != 0:
        query = query.filter(PopulationMetric.gender_id == filters["gender"])

    # Start year filter
    if filters.get("start_year") and filters["start_year"] != 0:
        start_year = Year.query.get(filters["start_year"])
        if start_year:
            start_year_value = (
                start_year.year
                if hasattr(start_year, "year")
                else start_year.year_value
            )

            if hasattr(Year, "year"):
                query = query.filter(Year.year >= start_year_value)
            else:
                query = query.filter(Year.year_value >= start_year_value)

    # End year filter
    if filters.get("end_year") and filters["end_year"] != 0:
        end_year = Year.query.get(filters["end_year"])
        if end_year:
            end_year_value = (
                end_year.year if hasattr(end_year, "year") else end_year.year_value
            )

            if hasattr(Year, "year"):
                query = query.filter(Year.year <= end_year_value)
            else:
                query = query.filter(Year.year_value <= end_year_value)

    # Metric type filter
    if filters.get("metric_type") and filters["metric_type"] != 0:
        query = query.filter(PopulationMetric.metric_id == filters["metric_type"])

    return query


@main_bp.route("/")
def index():
    """Render home page with population statistics.

    Returns:
        flask.Response: Rendered index template with statistics.
    """
    # Check if there's a database error
    db_error = current_app.config.get("DATABASE_ERROR")
    if db_error:
        flash(f"Database error: {db_error}", "danger")
        return render_template(
            "index.html",
            stats={
                "location_count": "N/A",
                "year_range": "N/A",
                "total_population": "N/A",
            },
            db_error=True,
        )

    # Get basic stats for the home page
    stats = {}

    try:
        stats["location_count"] = Location.query.count()

        # Handle both year and year_value attributes
        min_year = None
        max_year = None
        if hasattr(Year, "year"):
            min_year = db.session.query(func.min(Year.year)).scalar()
            max_year = db.session.query(func.max(Year.year)).scalar()
        else:
            min_year = db.session.query(func.min(Year.year_value)).scalar()
            max_year = db.session.query(func.max(Year.year_value)).scalar()

        stats["year_range"] = f"{min_year or 'N/A'} - {max_year or 'N/A'}"

        # Get total population from the latest year using London's data
        latest_year_id = db.session.query(func.max(Year.year_id)).scalar()
        population_metric = MetricType.query.filter_by(metric_name="population").first()

        # Find London location
        london_location = None
        if hasattr(Location, "area_name"):
            london_location = Location.query.filter(
                Location.area_name == "London"
            ).first()
        else:
            london_location = Location.query.filter(
                Location.location_name == "London"
            ).first()

        if latest_year_id and population_metric and london_location:
            population_query = db.session.query(func.sum(PopulationMetric.value))
            population_query = population_query.filter(
                PopulationMetric.year_id == latest_year_id,
                PopulationMetric.metric_id == population_metric.metric_id,
                PopulationMetric.location_id == london_location.location_id,
            )
            stats["total_population"] = population_query.scalar()

            # Format with commas
            if stats["total_population"]:
                stats["total_population"] = f"{int(stats['total_population']):,}"
            else:
                stats["total_population"] = "N/A"
        else:
            stats["total_population"] = "N/A"
    except Exception as e:
        flash(f"Error fetching statistics: {str(e)}", "danger")
        stats = {
            "location_count": "N/A",
            "year_range": "N/A",
            "total_population": "N/A",
        }

    return render_template("index.html", stats=stats)


@main_bp.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    """Render dashboard page with data visualizations.

    Returns:
        flask.Response: Rendered dashboard template with charts.
    """
    # Initialize filter form with choices
    form = FilterForm()
    choices = get_filter_choices()

    form.location.choices = choices["locations"]
    form.demographic.choices = choices["demographics"]
    form.gender.choices = choices["genders"]
    form.start_year.choices = choices["years"]
    form.end_year.choices = choices["years"]
    form.metric_type.choices = choices["metrics"]

    # Initialize filters dictionary with default values
    filters = {
        "location": 0,
        "demographic": 0,
        "gender": 0,
        "start_year": 0,
        "end_year": 0,
        "metric_type": 0,
    }

    # Process form submission
    if request.method == "POST":
        # Update filters with form data
        filters = {
            "location": int(request.form.get("location", 0)),
            "demographic": int(request.form.get("demographic", 0)),
            "gender": int(request.form.get("gender", 0)),
            "start_year": int(request.form.get("start_year", 0)),
            "end_year": int(request.form.get("end_year", 0)),
            "metric_type": int(request.form.get("metric_type", 0)),
        }

        # Update form with submitted values
        form.location.data = filters["location"]
        form.demographic.data = filters["demographic"]
        form.gender.data = filters["gender"]
        form.start_year.data = filters["start_year"]
        form.end_year.data = filters["end_year"]
        form.metric_type.data = filters["metric_type"]

        # Store filters in session
        session["dashboard_filters"] = filters
    elif "dashboard_filters" in session:
        # Use filters from session
        filters = session["dashboard_filters"]

        # Update form with session values
        form.location.data = filters.get("location", 0)
        form.demographic.data = filters.get("demographic", 0)
        form.gender.data = filters.get("gender", 0)
        form.start_year.data = filters.get("start_year", 0)
        form.end_year.data = filters.get("end_year", 0)
        form.metric_type.data = filters.get("metric_type", 0)

    # Generate charts based on filters
    charts = create_dashboard_charts(filters)

    # Convert charts to JSON for embedding in the template
    chart_json = {}
    for name, fig in charts.items():
        chart_json[name] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("dashboard.html", form=form, charts=chart_json)


@main_bp.route("/trends", methods=["GET", "POST"])
def trends():
    """Render trends and predictions page.

    Returns:
        flask.Response: Rendered trends template with prediction form.
    """
    # Initialize prediction form with choices
    form = PredictionForm()
    choices = get_filter_choices()

    # Remove "All" options for prediction form
    form.location.choices = (
        choices["locations"][1:]
        if len(choices["locations"]) > 1
        else choices["locations"]
    )
    form.demographic.choices = (
        choices["demographics"][1:]
        if len(choices["demographics"]) > 1
        else choices["demographics"]
    )
    form.gender.choices = (
        choices["genders"][1:] if len(choices["genders"]) > 1 else choices["genders"]
    )
    form.metric_type.choices = (
        choices["metrics"][1:] if len(choices["metrics"]) > 1 else choices["metrics"]
    )

    # Initialize variables for template
    prediction_chart = None
    confidence = None
    message = None

    # Process form submission
    if form.validate_on_submit():
        try:
            # Get form data
            location_id = form.location.data
            demographic_id = form.demographic.data
            gender_id = form.gender.data
            metric_id = form.metric_type.data
            years_ahead = form.years_ahead.data

            # Import here to avoid circular imports
            from population_app.ml import make_prediction

            # Get prediction
            fig_json, confidence = make_prediction(
                location_id, demographic_id, gender_id, metric_id, years_ahead
            )

            if fig_json is None:
                message = confidence  # Error message
            else:
                message = (
                    f"Prediction generated with {confidence} confidence. "
                    "The model has analyzed historical population data to "
                    "forecast future trends."
                )
                prediction_chart = fig_json

        except Exception as e:
            message = f"Error generating prediction: {str(e)}"

    return render_template(
        "trends.html",
        form=form,
        prediction_chart=prediction_chart,
        confidence=confidence,
        message=message,
    )


@main_bp.route("/data", methods=["GET", "POST"])
def data():
    """Render data filtering and export page.

    Returns:
        flask.Response: Rendered data template with filter and export forms.
    """
    # Initialize forms with choices
    filter_form = FilterForm()
    export_form = DataExportForm()
    choices = get_filter_choices()

    filter_form.location.choices = choices["locations"]
    filter_form.demographic.choices = choices["demographics"]
    filter_form.gender.choices = choices["genders"]
    filter_form.start_year.choices = choices["years"]
    filter_form.end_year.choices = choices["years"]
    filter_form.metric_type.choices = choices["metrics"]

    # Initialize filters dictionary and results
    filters = {}
    results = []

    # Handle filter form submission
    if request.method == "POST" and "submit" in request.form:
        if filter_form.validate_on_submit():
            # Update filters with form data
            filters = {
                "location": filter_form.location.data,
                "demographic": filter_form.demographic.data,
                "gender": filter_form.gender.data,
                "start_year": filter_form.start_year.data,
                "end_year": filter_form.end_year.data,
                "metric_type": filter_form.metric_type.data,
            }

            # Store filters in session
            session["data_filters"] = filters

            try:
                # Perform database query with filters
                query = (
                    db.session.query(
                        PopulationMetric,
                        Location,
                        Demographic,
                        Gender,
                        Year,
                        MetricType,
                    )
                    .join(
                        Location, PopulationMetric.location_id == Location.location_id
                    )
                    .join(
                        Demographic,
                        PopulationMetric.demographic_id == Demographic.demographic_id,
                    )
                    .join(Gender, PopulationMetric.gender_id == Gender.gender_id)
                    .join(Year, PopulationMetric.year_id == Year.year_id)
                    .join(
                        MetricType, PopulationMetric.metric_id == MetricType.metric_id
                    )
                )

                # Apply filters
                query = apply_filters(query, filters)

                # Limit to 100 for display
                query_results = query.limit(100).all()

                # Process results
                results = []
                for r in query_results:
                    # Handle different attribute names
                    location_name = (
                        r[1].area_name
                        if hasattr(r[1], "area_name")
                        else r[1].location_name
                    )
                    age_group = (
                        r[2].age_group if hasattr(r[2], "age_group") else str(r[2].age)
                    )
                    year_value = r[4].year if hasattr(r[4], "year") else r[4].year_value

                    results.append(
                        (
                            location_name,
                            age_group,
                            r[3].gender_type,
                            year_value,
                            r[5].metric_name,
                            r[0].value,
                        )
                    )

                flash(
                    f"Showing {len(results)}results per page. Use export for full results.",
                    "info",
                )

            except Exception as e:
                flash(f"Error querying data: {str(e)}", "danger")
        else:
            flash("Form validation failed. Please check your inputs.", "danger")

    # Handle export form submission
    elif request.method == "POST":
        try:
            export_format = request.form.get("export_format")
            if not export_format:
                flash("Please select an export format", "warning")
                return render_template(
                    "data.html",
                    filter_form=filter_form,
                    export_form=export_form,
                    results=results,
                )

            # Get filters from session or use empty dict
            filters = session.get("data_filters", {})

            # Prepare filter parameters
            location_ids = (
                [filters.get("location")] if filters.get("location") else None
            )
            demographic_ids = (
                [filters.get("demographic")] if filters.get("demographic") else None
            )
            gender_ids = [filters.get("gender")] if filters.get("gender") else None
            metric_ids = (
                [filters.get("metric_type")] if filters.get("metric_type") else None
            )

            # Get start/end year values
            start_year_value = None
            if filters.get("start_year"):
                start_year = Year.query.get(filters.get("start_year"))
                if start_year:
                    start_year_value = (
                        start_year.year
                        if hasattr(start_year, "year")
                        else start_year.year_value
                    )

            end_year_value = None
            if filters.get("end_year"):
                end_year = Year.query.get(filters.get("end_year"))
                if end_year:
                    end_year_value = (
                        end_year.year
                        if hasattr(end_year, "year")
                        else end_year.year_value
                    )

            # Build metric data query with filters
            metric_data_query = db.session.query(PopulationMetric.metric_data_id)

            # Apply location filter
            if location_ids and location_ids[0] != 0:
                metric_data_query = metric_data_query.filter(
                    PopulationMetric.location_id == location_ids[0]
                )

            # Apply demographic filter
            if demographic_ids and demographic_ids[0] != 0:
                metric_data_query = metric_data_query.filter(
                    PopulationMetric.demographic_id == demographic_ids[0]
                )

            # Apply gender filter
            if gender_ids and gender_ids[0] != 0:
                metric_data_query = metric_data_query.filter(
                    PopulationMetric.gender_id == gender_ids[0]
                )

            # Apply metric filter
            if metric_ids and metric_ids[0] != 0:
                metric_data_query = metric_data_query.filter(
                    PopulationMetric.metric_id == metric_ids[0]
                )

            # Apply year filters
            if start_year_value:
                year_ids_query = db.session.query(Year.year_id)
                if hasattr(Year, "year"):
                    year_ids_query = year_ids_query.filter(
                        Year.year >= start_year_value
                    )
                else:
                    year_ids_query = year_ids_query.filter(
                        Year.year_value >= start_year_value
                    )
                year_ids = [r[0] for r in year_ids_query.all()]
                if year_ids:
                    metric_data_query = metric_data_query.filter(
                        PopulationMetric.year_id.in_(year_ids)
                    )

            if end_year_value:
                year_ids_query = db.session.query(Year.year_id)
                if hasattr(Year, "year"):
                    year_ids_query = year_ids_query.filter(Year.year <= end_year_value)
                else:
                    year_ids_query = year_ids_query.filter(
                        Year.year_value <= end_year_value
                    )
                year_ids = [r[0] for r in year_ids_query.all()]
                if year_ids:
                    metric_data_query = metric_data_query.filter(
                        PopulationMetric.year_id.in_(year_ids)
                    )

            # Get filtered metric IDs
            metric_data_ids = [r[0] for r in metric_data_query.all()]

            # Now get the complete data with all joined information
            data = []
            if metric_data_ids:
                # Query in batches to avoid issues with large datasets
                batch_size = 1000
                for i in range(0, len(metric_data_ids), batch_size):
                    batch = metric_data_ids[i : i + batch_size]

                    # Get all data in one query with explicit joins
                    query = (
                        db.session.query(
                            PopulationMetric,
                            Location,
                            Demographic,
                            Gender,
                            Year,
                            MetricType,
                        )
                        .join(
                            Location,
                            PopulationMetric.location_id == Location.location_id,
                        )
                        .join(
                            Demographic,
                            PopulationMetric.demographic_id
                            == Demographic.demographic_id,
                        )
                        .join(Gender, PopulationMetric.gender_id == Gender.gender_id)
                        .join(Year, PopulationMetric.year_id == Year.year_id)
                        .join(
                            MetricType,
                            PopulationMetric.metric_id == MetricType.metric_id,
                        )
                        .filter(PopulationMetric.metric_data_id.in_(batch))
                    )

                    # Execute query
                    batch_results = query.all()

                    # Process results into dict format
                    for r in batch_results:
                        metric, location, demographic, gender, year, metric_type = r

                        # Handle different attribute names
                        location_name = (
                            location.area_name
                            if hasattr(location, "area_name")
                            else location.location_name
                        )

                        age_group = (
                            demographic.age_group
                            if hasattr(demographic, "age_group")
                            else str(demographic.age)
                        )

                        year_value = (
                            year.year if hasattr(year, "year") else year.year_value
                        )

                        data.append(
                            {
                                "Location": location_name,
                                "Age Group": age_group,
                                "Gender": gender.gender_type,
                                "Year": year_value,
                                "Metric": metric_type.metric_name,
                                "Value": (
                                    float(metric.value)
                                    if metric.value is not None
                                    else None
                                ),
                            }
                        )

            # Create DataFrame
            df = pd.DataFrame(data)

            if df.empty:
                flash("No data found for export with the current filters", "warning")
                return redirect(url_for("main.data"))

            # Create file in memory
            output = BytesIO()

            # Export based on format type
            if export_format == "csv":
                df.to_csv(output, index=False)
                mimetype = "text/csv"
                filename = "population_data.csv"
            elif export_format == "excel":
                df.to_excel(output, index=False)
                mimetype = (
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
                filename = "population_data.xlsx"
            elif export_format == "json":
                output.write(df.to_json(orient="records").encode("utf-8"))
                mimetype = "application/json"
                filename = "population_data.json"
            else:
                raise ValueError(f"Unsupported format: {export_format}")

            # Set the file pointer to the beginning
            output.seek(0)

            # Return file
            return send_file(
                output, mimetype=mimetype, as_attachment=True, download_name=filename
            )

        except Exception as e:
            flash(f"Error exporting data: {str(e)}", "danger")

    # Use stored filters if available
    if "data_filters" in session:
        filters = session.get("data_filters")

        # Update form with stored filters
        if filters:
            filter_form.location.data = filters.get("location", 0)
            filter_form.demographic.data = filters.get("demographic", 0)
            filter_form.gender.data = filters.get("gender", 0)
            filter_form.start_year.data = filters.get("start_year", 0)
            filter_form.end_year.data = filters.get("end_year", 0)
            filter_form.metric_type.data = filters.get("metric_type", 0)

            # If no results yet but we have filters, automatically query
            if not results:
                try:
                    # Get base query for population data
                    query = (
                        db.session.query(
                            PopulationMetric,
                            Location,
                            Demographic,
                            Gender,
                            Year,
                            MetricType,
                        )
                        .join(
                            Location,
                            PopulationMetric.location_id == Location.location_id,
                        )
                        .join(
                            Demographic,
                            PopulationMetric.demographic_id
                            == Demographic.demographic_id,
                        )
                        .join(Gender, PopulationMetric.gender_id == Gender.gender_id)
                        .join(Year, PopulationMetric.year_id == Year.year_id)
                        .join(
                            MetricType,
                            PopulationMetric.metric_id == MetricType.metric_id,
                        )
                    )

                    # Apply filters
                    query = apply_filters(query, filters)

                    # Limit to 100 for display
                    query_results = query.limit(100).all()

                    # Process results
                    results = []
                    for r in query_results:
                        # Get location name
                        location_name = (
                            r[1].area_name
                            if hasattr(r[1], "area_name")
                            else r[1].location_name
                        )

                        # Get demographic age group
                        age_group = (
                            r[2].age_group
                            if hasattr(r[2], "age_group")
                            else str(r[2].age)
                        )

                        # Get year value
                        year_value = (
                            r[4].year if hasattr(r[4], "year") else r[4].year_value
                        )

                        results.append(
                            (
                                location_name,
                                age_group,
                                r[3].gender_type,
                                year_value,
                                r[5].metric_name,
                                r[0].value,
                            )
                        )
                except Exception as e:
                    flash(f"Error loading data from stored filters: {str(e)}", "danger")

    return render_template(
        "data.html", filter_form=filter_form, export_form=export_form, results=results
    )


@main_bp.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    """Render chatbot interface page.

    Returns:
        flask.Response: Rendered chatbot template.
    """
    form = ChatbotForm()
    return render_template("chatbot.html", form=form)


@admin_bp.route("/")
def admin_index():
    """Render admin dashboard page.

    Returns:
        flask.Response: Rendered admin index template with statistics.
    """
    # Get some stats for the admin dashboard
    stats = {
        "total_records": PopulationMetric.query.count() or 0,
        "locations": Location.query.count() or 0,
        "years": Year.query.count() or 0,
        "predictions": 0,
        "chatbot_queries": ChatbotQuery.query.count() or 0,
        "recent_uploads": [],
    }

    # Create an empty form for the template
    from population_app.forms import DataUploadForm

    form = DataUploadForm()

    return render_template("admin/index.html", stats=stats, form=form)


@admin_bp.route("/upload", methods=["GET", "POST"])
def upload():
    """Render data upload page.

    Returns:
        flask.Response: Rendered upload template or redirect.
    """
    from werkzeug.utils import secure_filename
    import os

    form = DataUploadForm()

    if form.validate_on_submit():
        flash("Upload feature is a placeholder in this version", "info")
        return redirect(url_for("admin.admin_index"))

    return render_template("admin/upload.html", form=form)


@admin_bp.route("/predictions")
def predictions():
    """Render predictions management page.

    Returns:
        flask.Response: Rendered predictions template.
    """
    return render_template("admin/predictions.html", predictions=[])


@admin_bp.route("/chatbot-logs")
def chatbot_logs():
    """Render chatbot query logs page.

    Returns:
        flask.Response: Rendered chatbot logs template.
    """
    queries = ChatbotQuery.query.order_by(ChatbotQuery.timestamp.desc()).limit(20).all()
    return render_template("admin/chatbot_logs.html", queries=queries)
