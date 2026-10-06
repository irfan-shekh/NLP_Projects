# Toxic Comment Classifier - Django Web Application

A production-ready Django web application that uses machine learning to detect toxic comments in text. Built with a Random Forest classifier trained on the Toxic Comment Classification Dataset.

## Features

- **ML-Powered Detection**: Uses a trained Random Forest classifier to identify toxic content
- **Real-time Analysis**: Instant prediction with confidence scores
- **Modern UI**: Clean, responsive design with gradient backgrounds and smooth animations
- **Sample Data**: Built-in sample comments for easy testing
- **API Endpoint**: RESTful API for programmatic access
- **Form Validation**: User-friendly error messages and input validation
- **Mobile Responsive**: Works seamlessly on all device sizes

## Technology Stack

- **Backend**: Django 4.2+
- **Machine Learning**: scikit-learn, Random Forest Classifier
- **NLP**: NLTK for text preprocessing
- **Data Processing**: pandas, numpy
- **Model Serialization**: joblib
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript

## Project Structure

```
toxic_classifier/
├── manage.py                 # Django management script
├── requirements.txt          # Python dependencies
├── train_model.py           # Model training script
├── models/                  # Saved ML models
│   ├── toxic_model.pkl      # Trained Random Forest model
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── toxic_classifier/        # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   ├── templates/           # Project templates
│   └── static/              # Static files (CSS, JS)
└── classifier/              # Django app
    ├── __init__.py
    ├── admin.py
    ├── apps.py
    ├── forms.py
    ├── models.py
    ├── views.py
    ├── urls.py
    ├── migrations/
    └── templates/
        └── classifier/
            └── home.html    # Main UI template
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
cd d:\project_learning\Machine_Learning\Toxic+Comment+Classifier\toxic_classifier
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the ML Model

The application requires a trained model. Run the training script:

```bash
python train_model.py
```

This script will:
- Load the training data from `../Notebook/train.csv`
- Preprocess the text data
- Train a Random Forest classifier
- Save the model and vectorizer to the `models/` directory

**Note:** Ensure the training data (`train.csv`) exists in the `../Notebook/` directory relative to the project.

### Step 5: Run Django Migrations

```bash
python manage.py migrate
```

### Step 6: Start the Development Server

```bash
python manage.py runserver
```

The application will be available at: `http://127.0.0.1:8000/`

## Usage

### Web Interface

1. Open your browser and navigate to `http://127.0.0.1:8000/`
2. Enter a comment in the text area
3. Click "Analyze Comment"
4. View the prediction and confidence score

### Sample Comments

Use the built-in sample buttons to test the classifier:
- **Toxic Example 1**: "i killed an insect and ate it"
- **Toxic Example 2**: "You are stupid and should die"
- **Safe Example 1**: "Is this sentence a good one"
- **Safe Example 2**: "I love learning new things every day"

### API Endpoint

Send POST requests to `/api/predict/` for programmatic access:

```bash
curl -X POST http://127.0.0.1:8000/api/predict/ -d "comment=Your comment here"
```

**Response Format:**
```json
{
    "is_toxic": false,
    "confidence": 8.77,
    "comment": "Your comment here"
}
```

## Model Details

### Training Data

The model is trained on the Toxic Comment Classification Dataset, which contains:
- 159,571 comments
- 6 toxicity labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
- Balanced dataset with 5,000 toxic and 5,000 non-toxic comments

### Preprocessing

Text preprocessing includes:
- Removing alphanumeric characters
- Removing punctuation
- Converting to lowercase
- Removing newlines
- Removing non-ASCII characters

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Vectorization**: TF-IDF (unigrams)
- **Stopwords**: English stopwords removed
- **Test Accuracy**: ~95%

## Configuration

### Django Settings

Key settings in `toxic_classifier/settings.py`:

```python
DEBUG = True  # Set to False in production
SECRET_KEY = 'django-insecure-dev-key-change-in-production-xyz123'  # Change in production
ALLOWED_HOSTS = ['*']  # Update with your domain in production
```

### Model Paths

Models are saved in:
- `models/toxic_model.pkl` - Trained classifier
- `models/vectorizer.pkl` - TF-IDF vectorizer

## Troubleshooting

### Model Not Found Error

If you see "Model not found. Please train the model first.":
1. Ensure you've run `python train_model.py`
2. Check that `models/toxic_model.pkl` and `models/vectorizer.pkl` exist
3. Verify the training data path in `train_model.py`

### Training Data Not Found

If training fails with "Dataset not found":
1. Ensure `train.csv` exists in `../Notebook/` directory
2. Update the `DATA_PATH` in `train_model.py` if your data is elsewhere

### Import Errors

If you encounter import errors:
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check your Python version (3.8+ required)
3. Activate your virtual environment if using one

### Port Already in Use

If port 8000 is already in use:
```bash
python manage.py runserver 8001
```

## Development

### Adding New Features

1. Modify views in `classifier/views.py`
2. Update templates in `classifier/templates/classifier/`
3. Add new URLs in `classifier/urls.py`
4. Update forms in `classifier/forms.py`

### Retraining the Model

To retrain with new data:
1. Update the training script `train_model.py`
2. Run `python train_model.py`
3. The new model will automatically be used by the application

## Production Deployment

### Security Checklist

Before deploying to production:

1. **Change SECRET_KEY**: Generate a new secret key
   ```python
   python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```

2. **Set DEBUG = False** in settings.py

3. **Update ALLOWED_HOSTS** with your domain

4. **Use environment variables** for sensitive data

5. **Enable HTTPS** for secure connections

6. **Configure static files** serving (use whitenoise or serve via CDN)

### Deployment Platforms

Recommended platforms:
- **Heroku**: Easy deployment with PostgreSQL
- **PythonAnywhere**: Simple Django hosting
- **AWS**: Elastic Beanstalk or EC2
- **DigitalOcean**: Django Droplets

### Static Files

Collect static files before deployment:
```bash
python manage.py collectstatic
```

## API Documentation

### POST /api/predict/

Analyzes a comment for toxicity.

**Parameters:**
- `comment` (string, required): The comment text to analyze

**Returns:**
```json
{
    "is_toxic": boolean,
    "confidence": number,
    "comment": string
}
```

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -d "comment=This is a test comment"
```

## License

This project is for educational purposes. The training dataset is from the Toxic Comment Classification Challenge on Kaggle.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the code comments
3. Ensure all dependencies are correctly installed

## Acknowledgments

- Training data: Jigsaw Toxic Comment Classification Challenge (Kaggle)
- ML Framework: scikit-learn
- Web Framework: Django
- UI Framework: Bootstrap 5
