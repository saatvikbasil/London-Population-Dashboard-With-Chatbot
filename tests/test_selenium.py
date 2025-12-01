"""Selenium integration tests for population analysis application."""

import os
import time
import pytest
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


@pytest.fixture(scope="module")
def chrome_driver():
    """Set up Chrome driver with appropriate options.

    Returns:
        WebDriver: Configured Chrome WebDriver instance.
    """
    # Check if running on CI (GitHub Actions)
    options = ChromeOptions()
    if "GITHUB_ACTIONS" in os.environ:
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
    else:
        options.add_argument("--start-maximized")

    try:
        driver = Chrome(options=options)
        # Increase default timeout
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)

        yield driver
        driver.quit()
    except Exception as e:
        pytest.skip(f"Selenium setup failed: {str(e)}")


@pytest.fixture(scope="module")
def flask_server(app):
    """Start Flask server for testing.

    Args:
        app (Flask): Flask application to run.

    Yields:
        str: URL of the running Flask server.
    """
    # Use the test app config to run server
    port = 5001  # Use a different port than the default

    # Start the Flask server in a different thread
    import threading

    server_thread = threading.Thread(
        target=app.run,
        kwargs={
            "debug": False,
            "use_reloader": False,
            "host": "127.0.0.1",
            "port": port,
        },
    )
    server_thread.daemon = True
    server_thread.start()

    # Give the server a moment to start
    time.sleep(3)

    yield f"http://127.0.0.1:{port}"


class TestAppInterface:
    """Integration tests for the application interface."""

    def test_home_page_loads(self, chrome_driver, flask_server):
        """Test that the home page loads and contains key elements.

        Args:
            chrome_driver (WebDriver): Selenium WebDriver.
            flask_server (str): URL of the running Flask server.
        """
        chrome_driver.get(f"{flask_server}/")

        # Check the title - be more relaxed about exact matching
        assert "London" in chrome_driver.title, "Page title should contain 'London'"

        # Check for main elements - more flexible approach to finding headings
        headings = WebDriverWait(chrome_driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "h1, h2, h3"))
        )

        # Check if any heading contains expected text
        heading_texts = [h.text for h in headings]
        assert any(
            "London" in text for text in heading_texts
        ), f"No heading contains 'London'. Found: {heading_texts}"

        # Check for dashboard link - try different ways to find it
        try:
            # Try with link text
            dashboard_link = WebDriverWait(chrome_driver, 5).until(
                EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, "Dashboard"))
            )
        except TimeoutException:
            # Try with CSS selector for any link
            dashboard_links = chrome_driver.find_elements(
                By.CSS_SELECTOR, "a[href*='dashboard']"
            )
            assert len(dashboard_links) > 0, "No dashboard link found on the page"

    def test_dashboard_filters(self, chrome_driver, flask_server):
        """Test that the dashboard page loads and filters can be used.

        Args:
            chrome_driver (WebDriver): Selenium WebDriver.
            flask_server (str): URL of the running Flask server.
        """
        chrome_driver.get(f"{flask_server}/dashboard")

        # Wait for page to load
        WebDriverWait(chrome_driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Check for filter form - more reliable method
        try:
            filter_section = WebDriverWait(chrome_driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "form, .filter-panel, [id*='filter']")
                )
            )

            # Look for any dropdown/select element in the form
            dropdowns = chrome_driver.find_elements(By.TAG_NAME, "select")
            assert len(dropdowns) > 0, "No filter dropdowns found"

            # Try to find a submit/update button
            buttons = chrome_driver.find_elements(By.TAG_NAME, "button")
            update_button = None

            for button in buttons:
                if any(
                    text in button.text.lower()
                    for text in ["update", "apply", "filter"]
                ):
                    update_button = button
                    break

            if update_button:
                update_button.click()
                time.sleep(2)  # Wait for page to update

            # Check for chart elements
            charts = chrome_driver.find_elements(
                By.CSS_SELECTOR,
                ".chart, [id*='chart'], [id*='trend'], [id*='population']",
            )
            assert (
                len(charts) > 0
            ), "No charts found on dashboard after updating filters"

        except (TimeoutException, NoSuchElementException) as e:
            # Take a screenshot for debugging
            screenshot_path = "dashboard_error.png"
            chrome_driver.save_screenshot(screenshot_path)
            assert False, (
                f"Error finding dashboard elements: {str(e)}. "
                f"Screenshot saved to {screenshot_path}"
            )

    def test_data_export_interface(self, chrome_driver, flask_server):
        """Test the data export interface.

        Args:
            chrome_driver (WebDriver): Selenium WebDriver.
            flask_server (str): URL of the running Flask server.
        """
        chrome_driver.get(f"{flask_server}/data")

        # Wait for page to load
        WebDriverWait(chrome_driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        try:
            # Look for export-related content
            export_elements = chrome_driver.find_elements(
                By.CSS_SELECTOR,
                "[id*='export'], [class*='export'], h2, h3, h4, .card-header, .card-title",
            )

            export_element_found = False
            for elem in export_elements:
                if any(
                    text in elem.text.lower() for text in ["export", "download", "data"]
                ):
                    export_element_found = True
                    break

            assert export_element_found, "No export section found on data page"

            # Look for select/dropdown elements
            format_options = chrome_driver.find_elements(
                By.CSS_SELECTOR,
                "select, [id*='format'], [name*='format'], [class*='dropdown']",
            )
            assert len(format_options) > 0, "No format selection dropdown found"

            # Look for download/export buttons
            buttons = chrome_driver.find_elements(
                By.CSS_SELECTOR,
                "button, .btn, [type='submit'], a[href*='export'], a[href*='download']",
            )

            download_button_found = False
            for button in buttons:
                button_text = button.text.lower()
                if any(text in button_text for text in ["download", "export"]):
                    download_button_found = True
                    break

            assert download_button_found, "No download/export button found"

        except (TimeoutException, NoSuchElementException) as e:
            # Take a screenshot for debugging
            screenshot_path = "data_export_error.png"
            chrome_driver.save_screenshot(screenshot_path)
            assert False, (
                f"Error testing data export interface: {str(e)}. "
                f"Screenshot saved to {screenshot_path}"
            )

    def test_chatbot_interface(self, chrome_driver, flask_server):
        """Test the chatbot interface.

        Args:
            chrome_driver (WebDriver): Selenium WebDriver.
            flask_server (str): URL of the running Flask server.
        """
        chrome_driver.get(f"{flask_server}/chatbot")

        # Wait for page to load completely
        WebDriverWait(chrome_driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        try:
            # Verify we're on the right page
            assert "chatbot" in chrome_driver.current_url.lower(), "Not on chatbot page"

            # Wait until the page is fully loaded
            WebDriverWait(chrome_driver, 10).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "[id*='chat'], [class*='chat']")
                )
            )

            # Wait for the input form
            form = WebDriverWait(chrome_driver, 10).until(
                EC.presence_of_element_located((By.ID, "chatForm"))
            )

            # Find and interact with input field
            input_field = WebDriverWait(chrome_driver, 10).until(
                EC.element_to_be_clickable((By.ID, "userInput"))
            )

            # Check input field state
            assert input_field.is_displayed(), "Input field is not visible"
            assert input_field.is_enabled(), "Input field is not enabled"

            # Set input value via JavaScript
            chrome_driver.execute_script(
                "arguments[0].value = arguments[1];",
                input_field,
                "What is the population of London?",
            )

            # Verify input value
            value = chrome_driver.execute_script(
                "return arguments[0].value;", input_field
            )
            assert (
                "London" in value
            ), f"Failed to set input value, current value: {value}"

            # Submit form via JavaScript
            chrome_driver.execute_script("arguments[0].submit();", form)

            # Wait for UI update
            time.sleep(2)

            # Verify query appears in page
            page_text = chrome_driver.page_source.lower()
            assert (
                "london" in page_text
            ), "Query text not found in page after submission"

        except Exception as e:
            # Take a screenshot for debugging
            screenshot_path = "chatbot_error.png"
            chrome_driver.save_screenshot(screenshot_path)

            # Save page source for detailed debugging
            with open("chatbot_page_source.html", "w", encoding="utf-8") as f:
                f.write(chrome_driver.page_source)

            # Re-raise with more context
            raise Exception(
                f"Error testing chatbot interface: {str(e)}. "
                f"Screenshot saved to {screenshot_path}"
            ) from e
