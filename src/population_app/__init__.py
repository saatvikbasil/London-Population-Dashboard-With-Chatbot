"""Flask application factory for population analysis application."""

import os
from datetime import datetime
from flask import Flask
from flask_wtf.csrf import CSRFProtect


def clear_chat_history(app):
    """Clear all chat history when the app starts.

    Args:
        app (Flask): The Flask application instance.
    """
    from population_app.models import db, ChatbotQuery

    try:
        # Delete all records from the ChatbotQuery table
        db.session.query(ChatbotQuery).delete()
        db.session.commit()
    except Exception as e:
        # In a production setting, you might want to implement
        # more robust error handling
        pass


def intcomma(value):
    """Add commas to an integer for better readability.

    Args:
        value (int or None): The number to format.

    Returns:
        str: Formatted number with commas, or empty string if None.
    """
    if value is None:
        return ""
    try:
        value = int(value)
        return f"{value:,}"
    except (ValueError, TypeError):
        return value


def create_app(test_config=None):
    """Create and configure the Flask application.

    Args:
        test_config (dict, optional): Configuration for testing.
            Defaults to None.

    Returns:
        Flask: Configured Flask application instance.
    """
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # Configure the app
    app.config.from_mapping(
        SECRET_KEY="dev",
        SQLALCHEMY_DATABASE_URI=(
            "sqlite:///" + os.path.join(app.instance_path, "population.db")
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize CSRF protection
    csrf = CSRFProtect()
    csrf.init_app(app)

    # Initialize database
    from population_app.models import db

    db.init_app(app)

    # Register custom Jinja filters
    app.jinja_env.filters["intcomma"] = intcomma

    # Add template context processors
    @app.context_processor
    def inject_now():
        """Inject current datetime into all templates.

        Returns:
            dict: Current datetime.
        """
        return {"now": datetime.now()}

    with app.app_context():
        # Create database tables (schema only)
        db.create_all()

        # Import and register blueprints
        from population_app.routes import main_bp, admin_bp
        from population_app.api_routes import api_bp

        app.register_blueprint(main_bp)
        app.register_blueprint(admin_bp, url_prefix="/admin")
        app.register_blueprint(api_bp, url_prefix="/api")

        # Exempt API routes from CSRF protection
        csrf.exempt(api_bp)

        # Clear chat history on app start
        clear_chat_history(app)

        try:
            # Initialize database
            from population_app.helpers import init_database

            init_database()
        except Exception as e:
            # Track database initialization error
            app.config["DATABASE_ERROR"] = str(e)

    # Register error handlers
    from population_app.routes import handle_404, handle_500

    app.register_error_handler(404, handle_404)
    app.register_error_handler(500, handle_500)

    return app
