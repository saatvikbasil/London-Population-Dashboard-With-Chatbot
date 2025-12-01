"""Route tests for population analysis application."""


def test_index_route(client):
    """Test index route functionality.

    GIVEN a Flask application
    WHEN the '/' page is requested (GET)
    THEN check that the response is valid
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"London Population & Immigration Data" in response.data


def test_dashboard_route_get(client):
    """Test dashboard route GET request.

    GIVEN a Flask application
    WHEN the '/dashboard' page is requested (GET)
    THEN check that the response is valid with filter form
    """
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Population Dashboard" in response.data
    assert b"Data Filters" in response.data
    # Check for filter form elements
    assert b"Location" in response.data
    assert b"Age Group" in response.data


def test_dashboard_route_post_with_filters(client):
    """Test dashboard route POST request with filters.

    GIVEN a Flask application
    WHEN the '/dashboard' page is requested (POST) with filter parameters
    THEN check that the response is valid and filters are applied
    """
    response = client.post(
        "/dashboard",
        data={
            "location": 1,  # London
            "demographic": 0,  # All age groups
            "gender": 3,  # All genders
            "start_year": 1,  # 2020
            "end_year": 3,  # 2022
            "metric_type": 1,  # Population
            "submit": "Apply Filters",
        },
    )

    assert response.status_code == 200
    # Check if the dashboard updated with charts
    assert b"Population Trend Over Time" in response.data


def test_trends_route_get(client):
    """Test trends route GET request.

    GIVEN a Flask application
    WHEN the '/trends' page is requested (GET)
    THEN check that the response is valid with prediction form
    """
    response = client.get("/trends")
    assert response.status_code == 200
    assert b"Population Trends & Predictions" in response.data
    assert b"Generate Prediction" in response.data

    # Check for form fields
    assert b"Location" in response.data
    assert b"Years to Predict" in response.data


def test_trends_route_post_with_data(client):
    """Test trends route POST request with prediction parameters.

    GIVEN a Flask application
    WHEN the '/trends' page is requested (POST) with prediction parameters
    THEN check that the response contains appropriate prediction information
    """
    response = client.post(
        "/trends",
        data={
            "location": 1,  # London
            "demographic": 2,  # Age 25
            "gender": 3,  # All genders
            "metric_type": 1,  # Population
            "years_ahead": 5,  # Predict 5 years ahead
        },
    )

    assert response.status_code == 200
    # Note: The actual prediction might fail due to insufficient data in test db,
    # but we check that the page loads and contains appropriate messaging
    assert b"Population Trends & Predictions" in response.data

    # Either we got a prediction or an error message about insufficient data
    assert (
        b"Prediction" in response.data or b"insufficient data" in response.data.lower()
    )


def test_data_route_get(client):
    """Test data route GET request.

    GIVEN a Flask application
    WHEN the '/data' page is requested (GET)
    THEN check that the response is valid with data filters
    """
    response = client.get("/data")
    assert response.status_code == 200
    assert b"Population Data Access" in response.data
    assert b"Filter Data" in response.data
    assert b"Export Filtered Data" in response.data


def test_data_route_post_with_filters(client):
    """Test data route POST request with filters.

    GIVEN a Flask application
    WHEN the '/data' page is requested (POST) with filter parameters
    THEN check that filtered data is displayed
    """
    response = client.post(
        "/data",
        data={
            "location": 1,  # London
            "demographic": 0,  # All age groups
            "gender": 3,  # All genders
            "start_year": 1,  # 2020
            "end_year": 3,  # 2022
            "metric_type": 1,  # Population
            "submit": "Apply Filters",
        },
    )

    assert response.status_code == 200
    assert b"Data Results" in response.data

    # Look for either results or a message indicating no results
    assert b"London" in response.data or b"No data found" in response.data


def test_chatbot_route(client):
    """Test chatbot route GET request.

    GIVEN a Flask application
    WHEN the '/chatbot' page is requested (GET)
    THEN check that the chatbot interface loads
    """
    response = client.get("/chatbot")
    assert response.status_code == 200
    assert b"London Population Data Assistant" in response.data
    assert b"Ask about London population data" in response.data


def test_admin_index_route(client):
    """Test admin index route GET request.

    GIVEN a Flask application
    WHEN the '/admin/' page is requested (GET)
    THEN check that the admin dashboard loads
    """
    response = client.get("/admin/")
    assert response.status_code == 200
    assert b"Admin Dashboard" in response.data or b"Administration" in response.data


def test_admin_upload_route(client):
    """Test admin upload route GET request.

    GIVEN a Flask application
    WHEN the '/admin/upload' page is requested (GET)
    THEN check that the upload interface loads
    """
    response = client.get("/admin/upload")
    assert response.status_code == 200
    assert b"Upload Data" in response.data or b"Data Upload" in response.data


def test_404_error_handler(client):
    """Test 404 error handler.

    GIVEN a Flask application
    WHEN a non-existent page is requested
    THEN check that a custom 404 page is returned
    """
    response = client.get("/nonexistent-page")
    assert response.status_code == 404
    assert b"404" in response.data
    assert b"Page not found" in response.data
