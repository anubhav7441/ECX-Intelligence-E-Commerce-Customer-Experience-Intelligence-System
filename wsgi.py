"""Production entry point: gunicorn wsgi:app (Linux/Docker) or waitress-serve --listen=0.0.0.0:8000 wsgi:app (Windows)."""
from app import create_app

app = create_app()
