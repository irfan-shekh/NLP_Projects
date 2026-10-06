# Fake News Classifier 📰

A production-ready Django web application that uses Machine Learning to detect fake news with 89% accuracy. Built with modern UI, real-time predictions, and comprehensive error handling.

## 🌟 Features

- **🤖 ML-Powered Detection**: Naive Bayes classifier with TF-IDF vectorization
- **⚡ Real-time Analysis**: Instant predictions with confidence scores
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🎨 Modern UI**: Beautiful gradient-based interface with smooth animations
- **📊 Confidence Metrics**: Detailed prediction confidence for both fake and real classifications
- **🧪 Sample Testing**: Pre-loaded sample texts for easy testing
- **🔒 Error Handling**: Comprehensive error handling and user feedback

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fake-news-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML model**
   ```bash
   python train_model.py
   ```
   This will:
   - Download required NLTK data
   - Train the Naive Bayes model on 20,800 news samples
   - Save model artifacts (fake_news_model.joblib, tfidf_vectorizer.joblib, text_preprocessor.joblib)
   - Display model accuracy and test predictions

5. **Run Django migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:8000`

## 🏗️ Project Structure

```
fake-news-classifier/
├── fakenews_project/          # Django project settings
│   ├── settings.py           # Django configuration
│   ├── urls.py              # Main URL routing
│   └── wsgi.py              # WSGI configuration
├── classifier/               # Main Django app
│   ├── views.py             # Main views and ML integration
│   ├── urls.py              # App URL routing
│   └── templates/classifier/ # HTML templates
├── Notebook/                # Original Jupyter notebook
│   ├── Fake News Classifier.ipynb
│   └── nlp_utils.py
├── static/                  # Static files (CSS, JS, images)
├── templates/               # Django templates
├── train_model.py          # ML model training script
├── requirements.txt        # Python dependencies
├── manage.py              # Django management script
├── fake_news_model.joblib # Trained ML model
├── tfidf_vectorizer.joblib # Text vectorizer
├── text_preprocessor.joblib # Text preprocessing pipeline
└── README.md              # This file
```

## 🤖 Machine Learning Pipeline

### Data Processing

1. **Text Preprocessing**
   - Contraction expansion
   - Lowercase conversion
   - Punctuation and number removal
   - Tokenization
   - Stopword removal
   - Lemmatization with POS tagging

2. **Feature Extraction**
   - TF-IDF vectorization (max 5,000 features)
   - N-gram analysis (1-2 grams)
   - Minimum document frequency: 5
   - Maximum document frequency: 80%

3. **Model Training**
   - Algorithm: Multinomial Naive Bayes
   - Alpha parameter: 0.1 (Laplace smoothing)
   - Training samples: 20,800 news articles
   - Test split: 80/20 with stratification

### Model Performance

- **Accuracy**: 89.12%
- **Precision**: 87% (Fake), 91% (Real)
- **Recall**: 96% (Fake), 82% (Real)
- **F1-Score**: 91% (Fake), 86% (Real)

## 🎯 Usage Guide

### Web Interface

1. **Home Page**: Main prediction interface
   - Enter news text in the textarea
   - Click "Analyze News" to get prediction
   - View results with confidence scores

2. **Sample Testing**: Click on sample texts to test the classifier
3. **About Page**: Detailed information about the technology and methodology

### API Endpoint

**POST** `/predict/`

```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "prediction": "FAKE" | "REAL",
  "confidence": 85.67,
  "fake_confidence": 85.67,
  "real_confidence": 14.33,
  "text_length": 150,
  "processed_text": "processed text preview..."
}
```

## 🛠️ Technology Stack

### Backend
- **Django 4.2.7**: Web framework
- **scikit-learn 1.3.2**: Machine learning library
- **pandas 2.1.3**: Data manipulation
- **numpy 1.24.3**: Numerical computing
- **nltk 3.8.1**: Natural language processing
- **joblib 1.3.2**: Model serialization

### Frontend
- **Bootstrap 5.3.0**: UI framework
- **Font Awesome 6.4.0**: Icons
- **Google Fonts (Inter)**: Typography
- **Custom CSS**: Gradients, animations, responsive design

### Development
- **Python 3.8+**: Programming language
- **SQLite**: Database (development)
- **Whitenoise**: Static file serving

## 🔧 Configuration

### Django Settings

Key settings in `fakenews_project/settings.py`:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'classifier',  # Our app
]

ALLOWED_HOSTS = ['localhost', '127.0.0.1']

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'
```

### Model Configuration

Model parameters in `train_model.py`:

```python
# TF-IDF parameters
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8
)

# Naive Bayes parameters
model = MultinomialNB(alpha=0.1)
```

## 🧪 Testing

### Manual Testing

1. **Basic Functionality**
   - Enter real news text → Should predict "REAL"
   - Enter fake news text → Should predict "FAKE"
   - Test with short text → Should show validation error
   - Test with long text → Should work normally

2. **Sample Texts**
   - Click on provided sample texts
   - Verify predictions make sense
   - Check confidence scores

3. **Edge Cases**
   - Empty text → Error message
   - Very short text → Validation warning
   - Special characters → Should handle gracefully
   - Mixed case → Should work normally

### Automated Testing

```bash
# Test Django setup
python manage.py test

# Test model loading
python -c "
import joblib
model = joblib.load('fake_news_model.joblib')
print('Model loaded successfully')
"
```

## 🚀 Deployment

### Production Setup

1. **Environment Variables**
   ```bash
   export DEBUG=False
   export SECRET_KEY='your-secret-key'
   export ALLOWED_HOSTS='yourdomain.com'
   ```

2. **Static Files**
   ```bash
   python manage.py collectstatic
   ```

3. **Database**
   ```bash
   python manage.py migrate
   ```

4. **WSGI Server**
   ```bash
   gunicorn fakenews_project.wsgi:application
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python train_model.py
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "fakenews_project.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Error: Can't get attribute 'TextPreprocessor'
   ```
   **Solution**: Ensure `train_model.py` has been run successfully

2. **NLTK Data Missing**
   ```
   Error: Resource punkt not found
   ```
   **Solution**: Run `python train_model.py` to download NLTK data

3. **Static Files Not Loading**
   ```
   Error: 404 for static files
   ```
   **Solution**: Run `python manage.py collectstatic`

4. **Server Not Starting**
   ```
   Error: Port already in use
   ```
   **Solution**: Use different port: `python manage.py runserver 8001`

### Performance Issues

1. **Slow Predictions**
   - Check model file size (should be < 10MB)
   - Monitor memory usage
   - Consider model optimization for production

2. **Memory Issues**
   - Reduce TF-IDF max_features
   - Implement model caching
   - Use server with sufficient RAM

## 📈 Model Improvement

### Potential Enhancements

1. **Algorithm Upgrades**
   - Try LSTM/Transformer models
   - Ensemble methods
   - Deep learning approaches

2. **Feature Engineering**
   - Sentiment analysis
   - Named entity recognition
   - Readability scores

3. **Data Augmentation**
   - Increase training dataset
   - Add multilingual support
   - Include more diverse sources

4. **Performance Optimization**
   - Model quantization
   - Feature selection
   - Hyperparameter tuning

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

## 🙏 Acknowledgments

- Original dataset from [source]
- NLTK library for text processing
- Scikit-learn for machine learning
- Django framework for web development
- Bootstrap for UI components

---

**Built with ❤️ using Django and Machine Learning**
