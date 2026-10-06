# AI Chatbot - Intelligent Conversation Assistant

A production-ready web application featuring an intelligent chatbot powered by Machine Learning and Natural Language Processing. Built with Django and scikit-learn, this application provides a modern, responsive interface for real-time conversations.

## Features

- **Real-time Chat Interface**: Modern, responsive chat UI with smooth animations
- **ML-Powered Responses**: Uses TF-IDF vectorization and cosine similarity for intelligent responses
- **Natural Language Processing**: Advanced text preprocessing including lemmatization and contraction expansion
- **Model Training**: In-app model training with progress indicators
- **Sample Messages**: Quick-start buttons for easy testing
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Mobile Responsive**: Fully responsive design that works on all devices
- **Status Monitoring**: Real-time model status indicators

## Technology Stack

### Backend
- **Django 4.2+**: Web framework
- **scikit-learn 1.3+**: Machine learning library
- **NLTK 3.8+**: Natural language processing
- **pandas 2.0+**: Data manipulation
- **numpy 1.24+**: Numerical computing
- **joblib 1.3+**: Model serialization

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive functionality
- **Responsive Design**: Mobile-first approach
- **CSS Animations**: Smooth user interactions

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

5. **Train the chatbot model**
   ```bash
   python manage.py runserver
   ```
   Then visit `http://localhost:8000` and click "Train Model" button.

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

## Usage

### Training the Model
1. Ensure the `dialogs.txt` file is in the `chatbot_app/` directory
2. Click the "Train Model" button in the web interface
3. Wait for training to complete (status indicator will turn green)

### Chatting with the Bot
1. Type your message in the input field
2. Press Enter or click Send
3. Use sample message buttons for quick testing
4. The bot will respond based on trained conversation patterns

### API Endpoints

#### Chat Endpoint
- **URL**: `/chat/`
- **Method**: `POST`
- **Body**: `{"message": "your message here"}`
- **Response**: `{"response": "bot response", "status": "success"}`

#### Model Status
- **URL**: `/status/`
- **Method**: `GET`
- **Response**: `{"is_trained": true, "conversations_count": 100, "status": "success"}`

#### Train Model
- **URL**: `/train/`
- **Method**: `POST`
- **Response**: `{"message": "Model trained successfully", "conversations_loaded": 100, "status": "success"}`

## Project Structure

```
chatbot/
|-- chatbot_project/          # Django project settings
|   |-- __init__.py
|   |-- settings.py
|   |-- urls.py
|   |-- wsgi.py
|   |-- asgi.py
|
|-- chatbot_app/             # Main Django app
|   |-- __init__.py
|   |-- admin.py
|   |-- apps.py
|   |-- models.py
|   |-- views.py              # Main application logic
|   |-- urls.py               # App URL patterns
|   |-- ml_model.py           # Machine learning model
|   |-- nlp_utils.py          # NLP utilities
|   |-- dialogs.txt           # Training data
|   |-- templates/
|   |   |-- chatbot_app/
|   |   |   |-- home.html     # Main chat interface
|
|-- manage.py                 # Django management script
|-- requirements.txt          # Python dependencies
|-- README.md                # This file
```

## Model Architecture

### Text Preprocessing
1. **Contraction Expansion**: Converts contractions like "don't" to "do not"
2. **Tokenization**: Splits text into individual words
3. **Lemmatization**: Reduces words to their base form using POS tagging
4. **Lowercasing**: Converts all text to lowercase

### Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical vectors
- **N-gram Features**: Uses both unigrams and bigrams (1-2 grams)
- **Stop Word Removal**: Eliminates common English stop words
- **Max Features**: Limits vocabulary to 1000 most frequent terms

### Similarity Matching
- **Cosine Similarity**: Measures similarity between user input and training data
- **Threshold Filtering**: Only responds when similarity > 0.2
- **Fallback Responses**: Provides default responses for low similarity matches

## Training Data

The chatbot is trained using conversation pairs in `dialogs.txt`. The file should contain:
- Alternating user and bot messages
- One message per line
- UTF-8 encoding
- Format: user message followed by bot response

Example:
```
Hello
Hi there! How can I help you today?
What can you do?
I can chat with you and answer questions based on my training.
```

## Configuration

### Django Settings
Key settings in `chatbot_project/settings.py`:
- `DEBUG = True` for development
- `ALLOWED_HOSTS = ['localhost', '127.0.0.1']` for local access
- Static files configured for CSS/JS serving

### Model Parameters
Adjustable parameters in `ml_model.py`:
- `max_features`: Vocabulary size (default: 1000)
- `similarity_threshold`: Minimum similarity for responses (default: 0.2)
- `ngram_range`: N-gram range (default: (1, 2))

## Troubleshooting

### Common Issues

#### Model Not Training
- **Problem**: Training button not working
- **Solution**: Ensure `dialogs.txt` exists and is properly formatted
- **Check**: File permissions and encoding (should be UTF-8)

#### Server Not Starting
- **Problem**: Django server fails to start
- **Solution**: Check for port conflicts, try different port with `python manage.py runserver 8001`
- **Check**: All dependencies installed correctly

#### Bot Not Responding
- **Problem**: Bot gives generic responses
- **Solution**: Train the model using the Train Model button
- **Check**: Model status indicator shows trained status

#### NLTK Data Issues
- **Problem**: NLTK data not found errors
- **Solution**: The app automatically downloads required NLTK data
- **Manual**: Run `python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"`

### Performance Optimization

#### Memory Usage
- Limit `max_features` in TF-IDF vectorizer for large datasets
- Use model persistence to avoid retraining
- Consider using smaller datasets for testing

#### Response Time
- Optimize similarity threshold for faster responses
- Cache frequent responses
- Use efficient data structures

## Development

### Adding New Features
1. Update views in `chatbot_app/views.py`
2. Modify templates in `chatbot_app/templates/chatbot_app/`
3. Update URLs in `chatbot_app/urls.py`
4. Test thoroughly before deployment

### Customizing Responses
1. Edit `dialogs.txt` with new conversation pairs
2. Retrain the model using the Train Model button
3. Test new responses in the chat interface

### Extending ML Model
1. Modify `ml_model.py` for new algorithms
2. Update preprocessing in `nlp_utils.py`
3. Add new features to the vectorization pipeline

## Deployment

### Production Considerations
- Set `DEBUG = False` in settings
- Configure proper `ALLOWED_HOSTS`
- Use production-ready server (Gunicorn)
- Set up proper static file serving
- Configure database for production
- Implement security measures (HTTPS, CSRF protection)

### Environment Variables
Set these in production:
- `SECRET_KEY`: Django secret key
- `DEBUG`: Set to False
- `ALLOWED_HOSTS`: Production domain
- Database configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review the documentation
- Create an issue in the repository
- Contact the development team

---

**Note**: This is a demonstration application. For production use, ensure proper security measures, testing, and deployment practices are followed.
