"""Database initialization and utility functions for population data application.

This module provides functions for initializing the database, creating sample data,
and handling data export and filtering operations.
"""

import os
import sqlite3
from io import BytesIO
from typing import Dict, List, Union, Any

import shutil
from flask import current_app
import numpy as np
import pandas as pd

from flask import send_file

from population_app.models import (
    db,
    Location,
    Year,
    Demographic,
    Gender,
    MetricType,
    PopulationMetric,
)


def init_database() -> None:
    """Initialize database with detailed error handling."""
    try:
        # Ensure instance path exists
        os.makedirs(current_app.instance_path, exist_ok=True)
        
        # Path for the instance database
        instance_db_path = os.path.join(current_app.instance_path, 'population.db')
        
        # Possible source database paths
        possible_source_paths = [
            os.path.join(os.path.dirname(__file__), 'data', 'population.db'),
            os.path.join(os.path.dirname(__file__), '..', 'data', 'population.db'),
        ]
        
        source_db_path = None
        for path in possible_source_paths:
            if os.path.exists(path):
                source_db_path = path
                break
        
        if not source_db_path:
            raise FileNotFoundError("Could not locate source database file")
        
        # Copy the source database to the instance folder
        shutil.copy(source_db_path, instance_db_path)
        
        # Verify database connection and data
        conn = sqlite3.connect(instance_db_path)
        cursor = conn.cursor()
        
        # Check if key tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            raise ValueError("No tables found in the database")
        
        # Check data in a key table (e.g., Location)
        cursor.execute("SELECT COUNT(*) FROM Location")
        location_count = cursor.fetchone()[0]
        
        if location_count == 0:
            raise ValueError("No data found in Location table")
        
        conn.close()
        
    except Exception as e:
        # Log the full error details
        error_msg = f"Database initialization failed: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)


def import_all_locations() -> None:
    """Import all locations from the raw database file."""
    # Only run if we have fewer locations than expected
    if db.session.query(Location).count() < 33:
        try:
            # Connect to the raw SQLite file
            conn = sqlite3.connect("instance/population.db")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all locations from the database file
            cursor.execute("SELECT location_id, area_code, area_name FROM Location")
            raw_locations = cursor.fetchall()
            conn.close()

            # Clear existing locations if any
            db.session.query(Location).delete()

            # Import all locations
            for loc in raw_locations:
                location = Location(
                    location_id=loc["location_id"],
                    area_code=loc["area_code"],
                    area_name=loc["area_name"],
                )
                db.session.add(location)

            db.session.commit()
        except Exception:
            db.session.rollback()
            raise


def create_sample_data() -> None:
    """Create sample data for testing if database doesn't exist."""
    # First ensure all tables are created
    db.create_all()

    # Load boroughs dynamically from the database
    boroughs = [loc.area_name for loc in Location.query.all()]

    # Create sample locations
    for i, borough in enumerate(boroughs, 1):
        location = Location(location_id=i, area_name=borough)
        db.session.add(location)

    # Create sample years
    for i, year_val in enumerate(range(2010, 2023), 1):
        year = Year(year_id=i, year=year_val)
        db.session.add(year)

    # Create sample demographics (age groups)
    age_groups = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    for i, age in enumerate(age_groups, 1):
        demographic = Demographic(demographic_id=i, age=age)
        db.session.add(demographic)

    # Create sample genders
    genders = ["male", "female", "all"]
    for i, gender_type in enumerate(genders, 1):
        gender = Gender(gender_id=i, gender_type=gender_type)
        db.session.add(gender)

    # Create sample metric types
    metrics = [
        ("population", "base"),
        ("immigration", "migration"),
        ("emigration", "migration"),
        ("net_migration", "migration"),
    ]
    for i, (metric_name, category) in enumerate(metrics, 1):
        metric = MetricType(metric_id=i, metric_name=metric_name, category=category)
        db.session.add(metric)

    # Commit the seed data to create the IDs
    db.session.commit()

    # Create sample population metrics
    np.random.seed(42)  # For reproducibility

    metric_id_counter = 1
    population_metrics = []

    for location_id in range(1, min(6, len(boroughs)) + 1):  # Limit to first 6 boroughs
        for year_id in range(1, 6):  # Limit to 5 years
            for demographic_id in range(
                1, min(5, len(age_groups)) + 1
            ):  # Limit to first 5 age groups
                for gender_id in range(1, len(genders) + 1):
                    for metric_id in range(1, len(metrics) + 1):
                        # Skip some combinations to make data more realistic
                        if np.random.random() < 0.5:
                            continue

                        # Calculate base value depending on metric type
                        if metric_id == 1:  # Total Population
                            base_value = np.random.randint(5000, 50000)
                        elif metric_id == 2:  # Immigration
                            base_value = np.random.randint(500, 5000)
                        elif metric_id == 3:  # Emigration
                            base_value = np.random.randint(300, 3000)
                        else:  # Net Migration
                            base_value = np.random.randint(-1000, 2000)

                        # Add some trends over years
                        year_factor = 1 + (year_id - 1) * 0.02

                        # Adjust for age groups
                        if demographic_id in [1, 2, 3]:  # Younger populations
                            age_factor = 0.8
                        elif demographic_id in [9, 10]:  # Older populations
                            age_factor = 0.6
                        else:
                            age_factor = 1.0

                        # Adjust for gender
                        gender_factor = 2.0 if gender_id == 3 else 1.0

                        # Calculate final value with some randomness
                        value = base_value * year_factor * age_factor * gender_factor
                        value *= 1 + np.random.normal(0, 0.1)  # Add +/- 10% noise

                        population_metrics.append(
                            PopulationMetric(
                                metric_data_id=metric_id_counter,
                                location_id=location_id,
                                demographic_id=demographic_id,
                                gender_id=gender_id,
                                year_id=year_id,
                                metric_id=metric_id,
                                value=round(value),
                            )
                        )
                        metric_id_counter += 1

                        # Batch commit to avoid memory issues
                        if len(population_metrics) >= 100:
                            db.session.bulk_save_objects(population_metrics)
                            db.session.commit()
                            population_metrics = []

    # Commit any remaining metrics
    if population_metrics:
        db.session.bulk_save_objects(population_metrics)
        db.session.commit()


def get_filter_choices() -> Dict[str, List[tuple[int, str]]]:
    """Get choices for filter form dropdowns.

    Returns:
        Dict containing lists of filter choices for locations, demographics,
        genders, years, and metrics.
    """
    try:
        locations = [(l.location_id, l.area_name) for l in Location.query.all()]
        locations.insert(0, (0, "All Locations"))

        # Handle both age and age_group attributes
        demographics = []
        for d in Demographic.query.all():
            if hasattr(d, "age_group"):
                demographics.append((d.demographic_id, d.age_group))
            elif hasattr(d, "age"):
                demographics.append((d.demographic_id, str(d.age)))
        demographics.insert(0, (0, "All Age Groups"))

        genders = [(g.gender_id, g.gender_type) for g in Gender.query.all()]
        genders.insert(0, (0, "All Genders"))

        # Handle both year and year_value attributes
        years = []
        for y in Year.query.order_by(Year.year_id).all():
            if hasattr(y, "year_value"):
                years.append((y.year_id, y.year_value))
            elif hasattr(y, "year"):
                years.append((y.year_id, y.year))
        years.insert(0, (0, "All Years"))

        # Filter out migration-related metrics from dropdown
        excluded_metrics = [
            "internal_in",
            "internal_out",
            "internal_net",
            "international_in",
            "international_out",
            "international_net",
            "special_change",
        ]

        metrics = [
            (m.metric_id, m.metric_name)
            for m in MetricType.query.all()
            if m.metric_name not in excluded_metrics
        ]
        metrics.insert(0, (0, "All Metrics"))

        return {
            "locations": locations,
            "demographics": demographics,
            "genders": genders,
            "years": years,
            "metrics": metrics,
        }
    except Exception:
        # Return default empty choices if there's an error
        return {
            "locations": [(0, "All Locations")],
            "demographics": [(0, "All Age Groups")],
            "genders": [(0, "All Genders")],
            "years": [(0, "All Years")],
            "metrics": [(0, "All Metrics")],
        }


def apply_filters(query: Any, filters: Dict[str, Union[int, str]]) -> Any:
    """Apply filters to a query based on form data.

    Args:
        query: The SQLAlchemy query to apply filters to.
        filters: A dictionary of filter parameters.

    Returns:
        Filtered SQLAlchemy query.
    """
    if filters.get("location") and filters["location"] != 0:
        query = query.filter(PopulationMetric.location_id == filters["location"])

    if filters.get("demographic") and filters["demographic"] != 0:
        query = query.filter(PopulationMetric.demographic_id == filters["demographic"])

    if filters.get("gender") and filters["gender"] != 0:
        query = query.filter(PopulationMetric.gender_id == filters["gender"])

    if filters.get("start_year") and filters["start_year"] != 0:
        # Get the Year object
        start_year = Year.query.get(filters["start_year"])
        if start_year:
            # Correctly access year value depending on the attribute name
            start_year_value = (
                start_year.year
                if hasattr(start_year, "year")
                else start_year.year_value
            )

            # Apply filter with correct attribute name
            if hasattr(Year, "year"):
                query = query.filter(Year.year >= start_year_value)
            else:
                query = query.filter(Year.year_value >= start_year_value)

    if filters.get("end_year") and filters["end_year"] != 0:
        # Get the Year object
        end_year = Year.query.get(filters["end_year"])
        if end_year:
            # Correctly access year value depending on the attribute name
            end_year_value = (
                end_year.year if hasattr(end_year, "year") else end_year.year_value
            )

            # Apply filter with correct attribute name
            if hasattr(Year, "year"):
                query = query.filter(Year.year <= end_year_value)
            else:
                query = query.filter(Year.year_value <= end_year_value)

    if filters.get("metric_type") and filters["metric_type"] != 0:
        query = query.filter(PopulationMetric.metric_id == filters["metric_type"])

    return query


def export_data(query_results: List[Any], format_type: str) -> Any:
    """Export filtered data to the specified format.

    Args:
        query_results: SQLAlchemy query results containing PopulationMetric objects.
        format_type: String indicating the export format ('csv', 'excel', or 'json').

    Returns:
        Flask response object with the appropriate file for download.

    Raises:
        ValueError: If an unsupported format is provided.
    """
    data = []

    # Prepare the data with proper column names
    for result in query_results:
        try:
            # Check if this is a tuple from a join query
            if isinstance(result, tuple) and len(result) > 1:
                # Assuming result[0] is PopulationMetric and rest are related models
                metric = result[0]
                location = next((r for r in result if isinstance(r, Location)), None)
                demographic = next(
                    (r for r in result if isinstance(r, Demographic)), None
                )
                gender = next((r for r in result if isinstance(r, Gender)), None)
                year = next((r for r in result if isinstance(r, Year)), None)
                metric_type = next(
                    (r for r in result if isinstance(r, MetricType)), None
                )
            else:
                # Assuming it's a PopulationMetric with relationships loaded
                metric = result
                location = metric.location
                demographic = metric.demographic
                gender = metric.gender
                year = metric.year
                metric_type = metric.metric_type

            # Handle different attribute names appropriately
            location_name = (
                location.area_name
                if location and hasattr(location, "area_name")
                else "Unknown"
            )

            age_group = (
                demographic.age_group
                if demographic and hasattr(demographic, "age_group")
                else str(demographic.age) if demographic else "Unknown"
            )

            year_value = (
                year.year
                if year and hasattr(year, "year")
                else year.year_value if year else "Unknown"
            )

            # Create a row dict with all data
            row = {
                "Location": location_name,
                "Age Group": age_group,
                "Gender": gender.gender_type if gender else "Unknown",
                "Year": year_value,
                "Metric": metric_type.metric_name if metric_type else "Unknown",
                "Value": metric.value if metric else 0,
            }
            data.append(row)
        except Exception:
            # Continue with other results if there's an error processing one
            continue

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create BytesIO object for in-memory file
    output = BytesIO()

    # Export based on format type
    if format_type == "csv":
        df.to_csv(output, index=False)
        mimetype = "text/csv"
        filename = "population_data.csv"
    elif format_type == "excel":
        df.to_excel(output, index=False)
        mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "population_data.xlsx"
    elif format_type == "json":
        output.write(df.to_json(orient="records").encode("utf-8"))
        mimetype = "application/json"
        filename = "population_data.json"
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    # Set the file pointer to the beginning
    output.seek(0)

    # Use Flask's send_file to return the in-memory file
    return send_file(
        output, mimetype=mimetype, as_attachment=True, download_name=filename
    )
