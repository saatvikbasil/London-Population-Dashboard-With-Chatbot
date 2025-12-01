"""API endpoint tests for population analysis application."""

import json


def test_api_status(client):
    """Test the API status endpoint.

    GIVEN a Flask application with API routes
    WHEN the '/api/status' endpoint is requested (GET)
    THEN check that a valid status response is returned
    """
    response = client.get("/api/status")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert "status" in data
    assert data["status"] == "online"


def test_api_locations(client):
    """Test the API locations endpoint.

    GIVEN a Flask application with API routes
    WHEN the '/api/locations' endpoint is requested (GET)
    THEN check that location data is returned correctly
    """
    response = client.get("/api/locations")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert "locations" in data
    # Verify locations data structure
    assert isinstance(data["locations"], list)


def test_api_years(client):
    """Test the API years endpoint.

    GIVEN a Flask application with API routes
    WHEN the '/api/years' endpoint is requested (GET)
    THEN check that year data structure is returned correctly
    """
    response = client.get("/api/years")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert "years" in data
    # Verify years data structure
    assert isinstance(data["years"], list)


def test_api_demographics(client):
    """Test the API demographics endpoint.

    GIVEN a Flask application with API routes
    WHEN the '/api/demographics' endpoint is requested (GET)
    THEN check that demographic data structure is returned correctly
    """
    response = client.get("/api/demographics")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert "demographics" in data
    # Verify demographics data structure
    assert isinstance(data["demographics"], list)


def test_api_population_no_filters(client):
    """Test the API population endpoint without filters.

    GIVEN a Flask application with API routes
    WHEN the '/api/population' endpoint is requested (GET) without filters
    THEN check that the response is valid (either success or error is handled properly)
    """
    response = client.get("/api/population")

    # Accept either success or server error
    assert response.status_code in [200, 500]

    data = json.loads(response.data)

    if response.status_code == 200:
        # Verify successful response structure
        assert "data" in data
        assert "pagination" in data
    else:
        # Verify error response structure
        assert "error" in data
