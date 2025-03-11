from flask import Flask
from flask_cors import CORS
from routes.routes import patients

app = Flask(__name__)
CORS(app)

# Registrar el blueprint
app.register_blueprint(patients, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
