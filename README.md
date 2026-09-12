# Medicine Recommendation System — Fixed & Runnable Version

## Project structure

```text
Medicine_Recommendation_System_Fixed/
├── main.py
├── requirements.txt
├── run_windows.bat
├── run_linux_mac.sh
├── README.md
├── models/
│   └── svc.pkl
├── datasets/
│   ├── Training.csv
│   ├── Symptom-severity.csv
│   ├── symtoms_df.csv
│   ├── precautions_df.csv
│   ├── workout_df.csv
│   ├── description.csv
│   ├── medications.csv
│   └── diets.csv
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── contact.html
│   ├── developer.html
│   └── blog.html
└── static/
    └── img.png
```

## Run on Windows

1. Install Python 3.10/3.11/3.12.
2. Open this project folder in VS Code.
3. Run `run_windows.bat`, or manually:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

4. Open http://127.0.0.1:5000/
5. for mob open:https://medicine-recommendation-system-1-y8cl.onrender.com

## Run on Linux/macOS

```bash
chmod +x run_linux_mac.sh
./run_linux_mac.sh
```

## API for Postman

POST `http://127.0.0.1:5000/api/predict`

JSON body:

```json
{
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
}
```

The API returns the predicted disease plus description, precautions, medications, diet and workouts from the supplied project datasets.

## What was fixed

- Corrected the missing `datasets/`, `models/`, `templates/`, and `static/` directory structure.
- Changed relative file paths to paths based on `main.py`, so the app can be started from any working directory.
- Removed the fragile hard-coded symptom index dictionary and generated it from `Training.csv` in the same feature order used by the model.
- Added validation for empty/unknown symptoms instead of crashing with `KeyError`.
- Added a JSON `/api/predict` endpoint for Postman/mobile/React integration.
- Fixed voice recognition so the transcript is placed into the symptom input field.
- Added a symptom datalist for easier input.
- Kept the supplied SVC model and CSV data rather than retraining or replacing them.
- Added a clear educational-use medical disclaimer.
