from pathlib import Path

from flask import Flask, jsonify, render_template, request

from models.email_classifier import EmailModels, load_emails


DATA = Path(__file__).parent / "data" / "emails.csv"
MAX_TEXT = 10000


def create_app(data_path=DATA):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    models = EmailModels()
    models.train(load_emails(data_path))

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/api/classify")
    def classify():
        if not request.is_json:
            return jsonify(error="Send a JSON object containing email_text."), 415
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object containing email_text."), 400
        text = payload.get("email_text")
        if not isinstance(text, str) or not text.strip():
            return jsonify(error="Enter nonempty email text."), 400
        if len(text) > MAX_TEXT:
            return jsonify(error="Email text must be 10000 characters or fewer."), 400
        try:
            prediction = models.predict(text.strip())
        except ValueError as error:
            return jsonify(error=str(error)), 422
        return jsonify(prediction)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Request body must be 64 KiB or smaller."), 413

    @app.get("/api/health")
    def health():
        return jsonify(status="running", models_trained=True)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
