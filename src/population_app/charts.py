"""
This module contains functions to create charts for the population dashboard.
It uses Plotly for visualization and SQLAlchemy for database queries.
"""

import traceback
import plotly.express as px
import pandas as pd
from sqlalchemy import func, and_
from population_app.models import (
    db,
    Location,
    Year,
    Demographic,
    Gender,
    MetricType,
    PopulationMetric,
)


def create_dashboard_charts(filters=None):
    """
    Create charts for the dashboard based on applied filters.

    Args:
        filters (dict, optional): Dictionary containing filter parameters.
            Possible keys: location, demographic, gender, start_year,
            end_year, metric_type. Defaults to None.

    Returns:
        dict: Dictionary of plotly chart objects for the dashboard.
    """
    # Get base query for population data
    try:
        # Get filter values
        location_id = filters.get("location", 0) if filters else 0
        demographic_id = filters.get("demographic", 0) if filters else 0
        gender_id = filters.get("gender", 0) if filters else 0
        start_year_id = filters.get("start_year", 0) if filters else 0
        end_year_id = filters.get("end_year", 0) if filters else 0
        metric_id = filters.get("metric_type", 0) if filters else 0

        # Get latest year from database
        latest_year_id = db.session.query(func.max(Year.year_id)).scalar()
        latest_year = (
            db.session.query(Year.year).filter(Year.year_id == latest_year_id).scalar()
        )

        # Get actual values from IDs
        if start_year_id != 0:
            start_year_value = Year.query.get(start_year_id).year
        else:
            start_year_value = db.session.query(func.min(Year.year)).scalar()

        if end_year_id != 0:
            end_year_value = Year.query.get(end_year_id).year
        else:
            end_year_value = db.session.query(func.max(Year.year)).scalar()

        # Find London's location_id (used for filtering)
        london_location = Location.query.filter(Location.area_name == "London").first()
        london_id = london_location.location_id if london_location else None

        base_query = (
            db.session.query(
                Location.area_name,
                Year.year,
                Demographic.age,
                Gender.gender_type,
                MetricType.metric_name,
                func.sum(PopulationMetric.value).label("value"),
            )
            .outerjoin(
                PopulationMetric, Location.location_id == PopulationMetric.location_id
            )
            .join(Year, PopulationMetric.year_id == Year.year_id)
            .join(
                Demographic,
                PopulationMetric.demographic_id == Demographic.demographic_id,
            )
            .join(Gender, PopulationMetric.gender_id == Gender.gender_id)
            .join(MetricType, PopulationMetric.metric_id == MetricType.metric_id)
        )

        # Apply filters if provided
        filter_conditions = []

        # Handle location filter
        if location_id != 0:
            # Specific location selected
            filter_conditions.append(PopulationMetric.location_id == location_id)
        elif london_id is not None:
            # All locations selected - exclude the London aggregate
            filter_conditions.append(PopulationMetric.location_id != london_id)

        if demographic_id != 0:
            filter_conditions.append(PopulationMetric.demographic_id == demographic_id)

        if gender_id != 0:
            filter_conditions.append(PopulationMetric.gender_id == gender_id)

        if start_year_value:
            filter_conditions.append(Year.year >= start_year_value)

        if end_year_value:
            filter_conditions.append(Year.year <= end_year_value)

        # For non-migration charts, use basic population metrics
        population_filter_conditions = list(filter_conditions)
        if metric_id != 0:
            population_filter_conditions.append(PopulationMetric.metric_id == metric_id)
        else:
            # Default to population if no metric specified
            pop_metric = MetricType.query.filter(
                MetricType.metric_name == "population"
            ).first()
            if pop_metric:
                population_filter_conditions.append(
                    PopulationMetric.metric_id == pop_metric.metric_id
                )

        # Apply filters for population charts
        population_query = base_query
        if population_filter_conditions:
            population_query = population_query.filter(
                and_(*population_filter_conditions)
            )

        # Create empty dictionary to store charts
        charts = {}

        # 1. Population Trend Over Time Chart
        trend_query = population_query.group_by(Year.year).order_by(Year.year)
        trend_results = trend_query.all()

        if trend_results:
            trend_df = pd.DataFrame(
                [(r[1], r[5]) for r in trend_results], columns=["Year", "Population"]
            )

            trend_fig = {
                "data": [
                    {
                        "x": trend_df["Year"].tolist(),
                        "y": trend_df["Population"].tolist(),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Population",
                        "marker": {"size": 8, "color": "#1f77b4"},
                        "line": {"width": 3, "shape": "spline"},
                    }
                ],
                "layout": {
                    "title": "Population Trend Over Time",
                    "xaxis": {"title": "Year", "gridcolor": "#eee"},
                    "yaxis": {"title": "Population", "gridcolor": "#eee"},
                    "template": "plotly_white",
                    "plot_bgcolor": "rgba(255,255,255,0.9)",
                    "paper_bgcolor": "rgba(255,255,255,0.9)",
                    "hovermode": "x unified",
                },
            }
            charts["population_trend"] = trend_fig
        else:
            # Create empty chart
            charts["population_trend"] = {
                "data": [
                    {"x": [], "y": [], "type": "scatter", "mode": "lines+markers"}
                ],
                "layout": {"title": "No Data Available"},
            }

        # 2. Borough chart - only show latest year
        borough_query = population_query

        # Add filter for latest year
        if latest_year_id:
            borough_query = borough_query.filter(Year.year_id == latest_year_id)

        # Exclude the London aggregate
        if london_id is not None:
            borough_query = borough_query.filter(Location.location_id != london_id)

        borough_query = borough_query.group_by(Location.area_name).order_by(
            func.sum(PopulationMetric.value).desc().nullslast()
        )
        borough_results = borough_query.all()

        if borough_results:
            # Fill NaN values with 0 for boroughs with no metrics
            borough_df = pd.DataFrame(
                [(r[0], r[5] if r[5] is not None else 0) for r in borough_results],
                columns=["Borough", "Population"],
            )

            borough_fig = {
                "data": [
                    {
                        "x": borough_df["Borough"].tolist(),
                        "y": borough_df["Population"].tolist(),
                        "type": "bar",
                        "marker": {
                            "color": borough_df["Population"].tolist(),
                            "colorscale": "Viridis",
                        },
                    }
                ],
                "layout": {
                    "title": f"Population by Borough ({latest_year})",
                    "xaxis": {
                        "title": "",  # Remove the title to prevent overlap
                        "tickangle": -45,  # Rotate labels by 45 degrees
                        "automargin": True,  # Allow margins to adjust automatically
                    },
                    "yaxis": {"title": "Population", "gridcolor": "#eee"},
                    "template": "plotly_white",
                    "margin": {
                        "b": 150
                    },  # Add bottom margin to accommodate rotated labels
                    "plot_bgcolor": "rgba(255,255,255,0.9)",
                    "paper_bgcolor": "rgba(255,255,255,0.9)",
                    "hovermode": "closest",
                },
            }
            charts["borough_comparison"] = borough_fig
        else:
            charts["borough_comparison"] = {
                "data": [{"x": [], "y": [], "type": "bar"}],
                "layout": {"title": "No Data Available"},
            }

        # 3. Demographic Breakdown Chart - SIMPLIFIED APPROACH
        # Get all demographic data first to ensure we have data
        demo_results = (
            db.session.query(
                Demographic.age, func.sum(PopulationMetric.value).label("population")
            )
            .join(
                PopulationMetric,
                Demographic.demographic_id == PopulationMetric.demographic_id,
            )
            .join(Year, PopulationMetric.year_id == Year.year_id)
            .join(MetricType, PopulationMetric.metric_id == MetricType.metric_id)
        )

        # Apply basic filters
        if location_id != 0:
            demo_results = demo_results.filter(
                PopulationMetric.location_id == location_id
            )

        # Use latest year
        if latest_year_id:
            demo_results = demo_results.filter(Year.year_id == latest_year_id)

        # Use population metric
        pop_metric = MetricType.query.filter(
            MetricType.metric_name == "population"
        ).first()
        if pop_metric:
            demo_results = demo_results.filter(
                PopulationMetric.metric_id == pop_metric.metric_id
            )

        # Group and execute query
        demo_results = demo_results.group_by(Demographic.age).all()

        if demo_results and len(demo_results) > 0:
            # Convert to DataFrame
            raw_df = pd.DataFrame(
                [(age, pop) for age, pop in demo_results], columns=["Age", "Population"]
            )

            if len(raw_df) > 0:
                # Function to group ages into 5-year ranges
                def get_age_range(age):
                    """Group ages into 10-year ranges."""
                    start = (age // 10) * 10
                    end = start + 9
                    return f"{start}-{end}"

                # Apply grouping
                raw_df["Age Range"] = raw_df["Age"].apply(get_age_range)

                # Group by the age range and sum populations
                demo_df = raw_df.groupby("Age Range")["Population"].sum().reset_index()

                # Sort by age range
                demo_df["Sort Key"] = (
                    demo_df["Age Range"].str.split("-").str[0].astype(int)
                )
                demo_df = demo_df.sort_values("Sort Key")

                # Create nice color scale
                colors = px.colors.qualitative.Pastel

                demo_fig = {
                    "data": [
                        {
                            "labels": demo_df["Age Range"].tolist(),
                            "values": demo_df["Population"].tolist(),
                            "type": "pie",
                            "hole": 0.4,
                            "hoverinfo": "label+percent+value",
                            "textinfo": "percent",
                            "marker": {
                                "colors": colors[: len(demo_df)],
                                "line": {"color": "#fff", "width": 2},
                            },
                            "pull": [0.05]
                            * len(
                                demo_df
                            ),  # Pull slices out slightly for better visibility
                        }
                    ],
                    "layout": {
                        "title": "Population by Age Group",
                        "template": "plotly_white",
                        "showlegend": True,
                        "legend": {
                            "orientation": "h",
                            "yanchor": "bottom",
                            "y": -0.2,
                            "xanchor": "center",
                            "x": 0.5,
                        },
                        "plot_bgcolor": "rgba(255,255,255,0.9)",
                        "paper_bgcolor": "rgba(255,255,255,0.9)",
                        "margin": {"t": 50, "b": 80, "l": 50, "r": 50},
                    },
                }
                charts["demographic_breakdown"] = demo_fig
            else:
                charts["demographic_breakdown"] = {
                    "data": [
                        {
                            "labels": ["No Data"],
                            "values": [1],
                            "type": "pie",
                            "hole": 0.4,
                        }
                    ],
                    "layout": {"title": "No Age Group Data Available"},
                }
        else:
            charts["demographic_breakdown"] = {
                "data": [
                    {"labels": ["No Data"], "values": [1], "type": "pie", "hole": 0.4}
                ],
                "layout": {"title": "No Age Group Data Available"},
            }

        # 4. Gender Distribution Chart
        gender_query = population_query.filter(Gender.gender_type != "all").group_by(
            Gender.gender_type
        )
        gender_query = gender_query.filter(Year.year_id == latest_year_id)
        gender_results = gender_query.all()

        if gender_results:
            gender_df = pd.DataFrame(
                [(r[3], r[5]) for r in gender_results], columns=["Gender", "Population"]
            )

            gender_fig = {
                "data": [
                    {
                        "x": gender_df["Gender"].tolist(),
                        "y": gender_df["Population"].tolist(),
                        "type": "bar",
                        "marker": {
                            "color": ["#e377c2", "#1f77b4"]
                        },  # FLIPPED: Now pink for female, blue for male
                    }
                ],
                "layout": {
                    "title": "Population by Gender",
                    "xaxis": {"title": "Gender"},
                    "yaxis": {"title": "Population", "gridcolor": "#eee"},
                    "template": "plotly_white",
                    "plot_bgcolor": "rgba(255,255,255,0.9)",
                    "paper_bgcolor": "rgba(255,255,255,0.9)",
                },
            }
            charts["gender_distribution"] = gender_fig
        else:
            charts["gender_distribution"] = {
                "data": [{"x": [], "y": [], "type": "bar"}],
                "layout": {"title": "No Data Available"},
            }

        # 5. Migration Trends Chart - Enhanced styling
        migration_query = (
            db.session.query(
                Year.year,
                MetricType.metric_name,
                func.sum(PopulationMetric.value).label("value"),
            )
            .join(Year, PopulationMetric.year_id == Year.year_id)
            .join(MetricType, PopulationMetric.metric_id == MetricType.metric_id)
            .filter(
                MetricType.metric_name.in_(
                    ["internal_net", "international_net", "special_change"]
                )
            )
        )

        # Apply location filter
        if location_id != 0:
            migration_query = migration_query.filter(
                PopulationMetric.location_id == location_id
            )
        elif london_id is not None:
            migration_query = migration_query.filter(
                PopulationMetric.location_id != london_id
            )

        # Apply year range filters
        if start_year_value:
            migration_query = migration_query.filter(Year.year >= start_year_value)
        if end_year_value:
            migration_query = migration_query.filter(Year.year <= end_year_value)

        migration_results = (
            migration_query.group_by(Year.year, MetricType.metric_name)
            .order_by(Year.year)
            .all()
        )

        if migration_results and len(migration_results) > 0:
            migration_df = pd.DataFrame(
                [(r[0], r[1], r[2]) for r in migration_results],
                columns=["Year", "Metric", "Value"],
            )

            # Pivot to get separate columns for each migration metric
            migration_pivot = migration_df.pivot(
                index="Year", columns="Metric", values="Value"
            ).reset_index()

            # If columns don't exist, create them with zeros
            if "internal_net" not in migration_pivot.columns:
                migration_pivot["internal_net"] = 0
            if "international_net" not in migration_pivot.columns:
                migration_pivot["international_net"] = 0
            if "special_change" not in migration_pivot.columns:
                migration_pivot["special_change"] = 0

            # Calculate total net (sum of all migration types)
            migration_pivot["Total Net"] = (
                migration_pivot["internal_net"]
                + migration_pivot["international_net"]
                + migration_pivot["special_change"]
            )

            # Custom hover template
            hovertemplate = (
                "<b>%{x}</b><br>"
                + "Value: %{y:,.0f}<br>"
                + "<extra>%{fullData.name}</extra>"
            )

            migration_fig = {
                "data": [
                    {
                        "x": migration_pivot["Year"].tolist(),
                        "y": migration_pivot["internal_net"].tolist(),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Internal Net Migration",
                        "line": {
                            "color": "#2ca02c",
                            "width": 2,
                            "shape": "spline",
                            "dash": "solid",
                        },
                        "marker": {"size": 8, "symbol": "circle"},
                        "hovertemplate": hovertemplate,
                    },
                    {
                        "x": migration_pivot["Year"].tolist(),
                        "y": migration_pivot["international_net"].tolist(),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "International Net Migration",
                        "line": {
                            "color": "#d62728",
                            "width": 2,
                            "shape": "spline",
                            "dash": "solid",
                        },
                        "marker": {"size": 8, "symbol": "circle"},
                        "hovertemplate": hovertemplate,
                    },
                    {
                        "x": migration_pivot["Year"].tolist(),
                        "y": migration_pivot["Total Net"].tolist(),
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Total Net Migration",
                        "line": {
                            "color": "#1f77b4",
                            "width": 3,
                            "shape": "spline",
                            "dash": "solid",
                        },
                        "marker": {"size": 10, "symbol": "circle"},
                        "hovertemplate": hovertemplate,
                    },
                ],
                "layout": {
                    "title": "Migration Trends",
                    "xaxis": {
                        "title": "Year",
                        "showgrid": True,
                        "gridcolor": "#eee",
                        "zeroline": True,
                        "zerolinecolor": "#444",
                        "zerolinewidth": 1,
                    },
                    "yaxis": {
                        "title": "Net Migration",
                        "showgrid": True,
                        "gridcolor": "#eee",
                        "zeroline": True,
                        "zerolinecolor": "#444",
                        "zerolinewidth": 1,
                        "tickformat": ",d",  # Format with commas for thousands
                    },
                    "template": "plotly_white",
                    "legend": {
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": -0.2,
                        "xanchor": "center",
                        "x": 0.5,
                        "bgcolor": "rgba(255,255,255,0.9)",
                        "bordercolor": "#ddd",
                        "borderwidth": 1,
                    },
                    "hovermode": "x unified",
                    "plot_bgcolor": "rgba(255,255,255,0.9)",
                    "paper_bgcolor": "rgba(255,255,255,0.9)",
                    "margin": {"t": 50, "b": 100, "l": 60, "r": 30},
                },
            }
            charts["migration_trends"] = migration_fig
        else:
            # Create empty chart with proper axes and labels
            charts["migration_trends"] = {
                "data": [
                    {
                        "x": [],
                        "y": [],
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Internal Net Migration",
                    },
                    {
                        "x": [],
                        "y": [],
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "International Net Migration",
                    },
                    {
                        "x": [],
                        "y": [],
                        "type": "scatter",
                        "mode": "lines+markers",
                        "name": "Total Net Migration",
                    },
                ],
                "layout": {
                    "title": "No Migration Data Available",
                    "xaxis": {"title": "Year"},
                    "yaxis": {"title": "Net Migration"},
                    "template": "plotly_white",
                },
            }

        return charts

    except Exception as e:

        traceback.print_exc()
        # Return empty charts on error
        empty_chart = {
            "data": [{"x": [], "y": [], "type": "scatter"}],
            "layout": {"title": f"Error: {str(e)}"},
        }
        return {
            "population_trend": empty_chart,
            "borough_comparison": empty_chart,
            "demographic_breakdown": empty_chart,
            "gender_distribution": empty_chart,
            "migration_trends": empty_chart,
        }
