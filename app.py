from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import hashlib
import traceback
import random

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Secret key for sessions
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')

# Session configuration
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_TYPE'] = 'filesystem'

# -----------------------------
# Database Configuration
# -----------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///spamshield.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -----------------------------
# Initialize OpenAI with better error handling
# -----------------------------
print("\n" + "="*50)
print("🔧 OPENAI INITIALIZATION")
print("="*50)

# Check if API key exists
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✅ OPENAI_API_KEY found: {api_key[:8]}...{api_key[-4:]}")
else:
    print("❌ OPENAI_API_KEY not found in .env file")
    print("   Please create a .env file with: OPENAI_API_KEY=your-key-here")

try:
    client = OpenAI(api_key=api_key)
    # Test the connection with a simple call
    test_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say 'test'"}],
        max_tokens=5
    )
    openai_available = True
    print("✅ OpenAI client initialized and tested successfully!")
    print(f"   Test response: {test_response.choices[0].message.content}")
except Exception as e:
    print(f"❌ OpenAI initialization failed: {type(e).__name__}")
    print(f"   Error details: {str(e)}")
    print(f"   Traceback: {traceback.format_exc()}")
    client = None
    openai_available = False

print("="*50 + "\n")

# Simple cache for AI explanations to avoid repeated API calls
explanation_cache = {}

# -----------------------------
# Load ML Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Load model with error handling
try:
    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None

# -----------------------------
# Database Models
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    country = db.Column(db.String(50), nullable=False)
    newsletter = db.Column(db.Boolean, default=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id} - {self.result}>'

# -----------------------------
# AI Explanation Function with Fallback
# -----------------------------
def generate_explanation(message, result):
    """
    Generate explanation - uses fallback when OpenAI is unavailable
    """
    # Create a cache key based on message and result
    cache_key = hashlib.md5(f"{message}_{result}".encode()).hexdigest()
    
    # Check cache first
    if cache_key in explanation_cache:
        print("✅ Using cached explanation")
        return explanation_cache[cache_key]
    
    # Check if OpenAI is available (it's not due to quota)
    if not openai_available or client is None:
        print("🤖 Using fallback explanation system (OpenAI unavailable)")
        
        # Return helpful mock explanations based on result
        if result == "Spam":
            explanations = [
                "⚠️ This message was flagged as spam because it contains urgent language and suspicious links. Always verify before clicking any links in unexpected messages.",
                "🚨 Our system detected this as spam due to promotional keywords and unusual sender patterns. Be cautious with messages promising prizes or asking for personal information.",
                "🔍 This appears to be spam - it uses pressure tactics like 'limited time' or 'act now' and may be trying to collect your personal data.",
                "📧 This message was flagged as spam because it contains phrases commonly used in phishing attempts. Never share passwords or financial details via email.",
                "🛡️ Spam detected! This message has characteristics of bulk promotional content or scam attempts. When in doubt, contact the company directly through their official website.",
                "⚠️ This message shows classic spam patterns: generic greeting, urgent call-to-action, and suspicious links. Delete and block the sender.",
                "🚩 Red flags detected: This message tries to create a false sense of urgency and asks you to click on untrusted links. Legitimate companies don't do this.",
                "📢 Promotional spam detected. This appears to be unsolicited marketing content. Mark as spam to train your filter.",
                "🔐 Security warning: This message contains elements typical of phishing scams. Do not reply or click any links.",
                "⚠️ Spam alert: The message uses emotional manipulation and offers unrealistic rewards. This is a common scam tactic."
            ]
        else:
            explanations = [
                "✅ This message appears legitimate. It contains personal language and specific details that automated spam typically lacks. Always stay cautious with attachments though.",
                "📨 Our model classified this as safe - it has characteristics of normal personal communication like proper grammar and contextual relevance.",
                "👍 This doesn't look like spam. It has the natural flow and specific references that legitimate messages contain, unlike mass-produced spam.",
                "💬 This appears to be a genuine message. It addresses you specifically and doesn't contain the urgent calls-to-action common in spam.",
                "🔐 Safe message detected. It lacks the suspicious patterns typical of spam, such as misspellings, excessive punctuation, or shady links.",
                "✅ Legitimate message: This contains personal context and specific details that spammers wouldn't know. Still, always verify unexpected requests.",
                "📧 This appears to be a normal communication from a real person. The language is natural and conversational, unlike automated spam.",
                "👍 No spam patterns detected. The message has proper formatting and doesn't try to pressure you into immediate action.",
                "💼 This looks like legitimate business or personal communication. It contains specific references that spammers couldn't fake.",
                "✅ Safe: The message doesn't contain any of the usual spam indicators like suspicious links, urgent requests, or grammatical errors."
            ]
        
        # Deterministic selection based on message hash for consistency
        idx = int(hashlib.md5(message.encode()).hexdigest(), 16) % len(explanations)
        explanation = explanations[idx]
        
        # Cache the result
        explanation_cache[cache_key] = explanation
        
        # Limit cache size to prevent memory issues
        if len(explanation_cache) > 200:
            # Remove oldest 50 items
            for _ in range(50):
                explanation_cache.pop(next(iter(explanation_cache)))
        
        return explanation
    
    # If OpenAI becomes available (after adding billing), use it
    try:
        print(f"\n🤖 Generating AI explanation for message: {message[:50]}...")
        
        # Different prompts based on result
        if result == "Spam":
            prompt = f"""
            A message was classified as: SPAM
            
            Message: "{message}"
            
            Please provide a brief analysis (2-3 sentences) explaining:
            1. Why this message appears to be spam
            2. What specific red flags it contains
            3. One practical safety tip for the user
            
            Keep it helpful and educational, not scary.
            """
        else:
            prompt = f"""
            A message was classified as: NOT SPAM (Legitimate)
            
            Message: "{message}"
            
            Please provide a brief analysis (2-3 sentences) explaining:
            1. Why this message appears legitimate
            2. What characteristics make it safe
            3. When users should still be cautious
            
            Keep it helpful and reassuring.
            """
        
        # Make API call with timeout
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=[
                {"role": "system", "content": "You are a helpful cybersecurity expert helping users understand spam detection. Keep responses concise and practical."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            timeout=30
        )
        
        explanation = response.choices[0].message.content
        print(f"✅ AI explanation generated successfully")
        
        # Cache the result
        explanation_cache[cache_key] = explanation
        
        # Limit cache size
        if len(explanation_cache) > 200:
            for _ in range(50):
                explanation_cache.pop(next(iter(explanation_cache)))
        
        return explanation
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"❌ OpenAI API error: {error_type} - {error_msg}")
        
        # Fallback to mock explanations on error
        if result == "Spam":
            return "⚠️ This message was flagged as spam by our ML model. For detailed AI-powered analysis, please ensure your OpenAI account has available credits."
        else:
            return "✅ This message appears legitimate according to our ML model. For detailed AI-powered analysis, please ensure your OpenAI account has available credits."

# -----------------------------
# Debug route to test OpenAI
# -----------------------------
@app.route('/test-openai')
def test_openai():
    """Test route to check OpenAI functionality"""
    if not openai_available:
        return {
            'status': 'info',
            'message': 'OpenAI not available - using fallback explanations',
            'api_key_present': bool(os.getenv('OPENAI_API_KEY')),
            'api_key_prefix': os.getenv('OPENAI_API_KEY', '')[:8] if os.getenv('OPENAI_API_KEY') else None,
            'using_fallback': True
        }
    
    try:
        test_message = "Test message"
        test_result = "Not Spam"
        explanation = generate_explanation(test_message, test_result)
        return {
            'status': 'success',
            'message': 'OpenAI working',
            'explanation': explanation,
            'api_key_present': True,
            'using_fallback': False
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'api_key_present': True,
            'using_fallback': True
        }

# -----------------------------
# PUBLIC ROUTES
# -----------------------------

@app.route('/')
def home():
    """Landing page - accessible to everyone"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page with all fields"""
    if request.method == 'POST':
        # Get all form data
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        username = request.form['username']
        email = request.form['email']
        phone = request.form.get('phone', '')
        country = request.form['country']
        newsletter = True if request.form.get('newsletter') else False
        password = request.form['password']
        
        # Validation
        errors = []
        
        # Check first name
        if not first_name or len(first_name) < 2:
            errors.append("First name must be at least 2 characters")
        
        # Check last name
        if not last_name or len(last_name) < 2:
            errors.append("Last name must be at least 2 characters")
        
        # Check username format
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters")
        if not username.isalnum() and '_' not in username:
            errors.append("Username can only contain letters, numbers, and underscore")
        
        # Check email format
        if not email or '@' not in email or '.' not in email:
            errors.append("Please enter a valid email address")
        
        # Check country selected
        if not country or country == "":
            errors.append("Please select your country")
        
        # Check password strength
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        if not any(c in '!@#$%^&*' for c in password):
            errors.append("Password must contain at least one special character (!@#$%^&*)")
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return redirect(url_for('register'))

        # Check if username exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists! Please choose another.', 'error')
            return redirect(url_for('register'))
        
        # Check if email exists
        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered! Please use another or login.', 'error')
            return redirect(url_for('register'))

        # Hash password and create user
        hashed_password = generate_password_hash(password)
        new_user = User(
            first_name=first_name,
            last_name=last_name,
            username=username,
            email=email,
            phone=phone,
            country=country,
            newsletter=newsletter,
            password=hashed_password
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('🎉 Registration successful! Please login to continue.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page - public"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            # Set session variables
            session.permanent = True
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_email'] = user.email
            session['first_name'] = user.first_name
            session['last_name'] = user.last_name
            session['logged_in'] = True
            
            flash(f'✨ Welcome back, {user.first_name}!', 'success')
            # Redirect to analyze page instead of home
            return redirect(url_for('analyze_page'))
        else:
            flash('❌ Invalid username or password!', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout - clears session"""
    session.clear()
    flash('👋 You have been logged out successfully!', 'success')
    return redirect(url_for('home'))

# -----------------------------
# Messaging Page (after login)
# -----------------------------
@app.route('/analyze-page')
def analyze_page():
    """Main messaging page where users can type and see results"""
    if 'user_id' not in session:
        flash('🔒 Please login to access the analyzer!', 'error')
        return redirect(url_for('login'))
    
    return render_template('messaging.html', 
                         username=session.get('username'),
                         openai_available=openai_available)

# -----------------------------
# Analysis route (handles form submission)
# -----------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze message and show results on the same page"""
    if 'user_id' not in session:
        flash('🔒 Please login to make predictions!', 'error')
        return redirect(url_for('login'))

    message = request.form['message']
    
    if not message or len(message.strip()) == 0:
        flash('Please enter a message to analyze!', 'error')
        return redirect(url_for('analyze_page'))
    
    # Check if model is loaded
    if model is None or vectorizer is None:
        flash('Model not available. Please contact administrator.', 'error')
        return redirect(url_for('analyze_page'))
    
    try:
        # Make prediction with ML model
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0]

        confidence = round(np.max(probability) * 100, 2)
        result = "Spam" if prediction == 1 else "Not Spam"

        # Generate AI explanation
        explanation = generate_explanation(message, result)

        # Save to database with explanation
        new_prediction = Prediction(
            message=message,
            result=result,
            confidence=confidence,
            explanation=explanation,
            user_id=session['user_id']
        )

        db.session.add(new_prediction)
        db.session.commit()
        
        # Render the messaging page with results
        return render_template('messaging.html', 
                             username=session.get('username'),
                             message=message,
                             result=result,
                             confidence=confidence,
                             explanation=explanation,
                             openai_available=openai_available)
    
    except Exception as e:
        flash(f'Error during prediction: {str(e)}', 'error')
        return redirect(url_for('analyze_page'))

# -----------------------------
# LEGAL PAGE ROUTES
# -----------------------------

@app.route('/terms-of-service')
def terms_of_service():
    """Terms of Service page"""
    return render_template('legal.html', 
                         title="Terms of Service",
                         last_updated="January 1, 2024",
                         content=[
                             {
                                 "title": "1. Acceptance of Terms",
                                 "content": "By accessing and using SpamShield AI, you agree to be bound by these Terms of Service."
                             },
                             {
                                 "title": "2. Use License",
                                 "content": "Permission is granted to temporarily use SpamShield AI for personal, non-commercial use."
                             }
                         ])

@app.route('/privacy-policy')
def privacy_policy():
    """Privacy Policy page"""
    return render_template('legal.html', 
                         title="Privacy Policy",
                         last_updated="January 1, 2024",
                         content=[
                             {
                                 "title": "1. Information We Collect",
                                 "content": "We collect information you provide directly to us when you create an account."
                             },
                             {
                                 "title": "2. How We Use Your Information",
                                 "content": "We use the information to provide and improve our spam detection services."
                             }
                         ])

# -----------------------------
# PROTECTED ROUTES
# -----------------------------

@app.route('/predict', methods=['POST'])
def predict():
    """Legacy prediction route - redirects to new analyze page"""
    if 'user_id' not in session:
        flash('🔒 Please login to make predictions!', 'error')
        return redirect(url_for('login'))
    
    message = request.form['message']
    # Redirect to analyze with the message
    return redirect(url_for('analyze_page'))

@app.route('/dashboard')
def dashboard():
    """User dashboard - requires login"""
    if 'user_id' not in session:
        flash('🔒 Please login to view dashboard!', 'error')
        return redirect(url_for('login'))

    # Get user's predictions
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
    
    # Calculate stats
    total = len(predictions)
    spam_count = sum(1 for p in predictions if p.result == "Spam")
    ham_count = sum(1 for p in predictions if p.result == "Not Spam")
    avg_confidence = round(sum(p.confidence for p in predictions) / total, 1) if total > 0 else 0
    
    # Get user info
    user = User.query.get(session['user_id'])
    
    return render_template('dashboard.html', 
                         predictions=predictions, 
                         username=session.get('username'),
                         spam_count=spam_count,
                         ham_count=ham_count,
                         total=total,
                         avg_confidence=avg_confidence,
                         member_since=user.created_at.strftime('%B %Y') if user else 'N/A')

@app.route('/delete-history', methods=['POST'])
def delete_history():
    """Delete user's prediction history"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        Prediction.query.filter_by(user_id=session['user_id']).delete()
        db.session.commit()
        flash('🗑️ Your prediction history has been cleared!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Error clearing history. Please try again.', 'error')
    
    return redirect(url_for('dashboard'))

# -----------------------------
# NEW: Delete Single Prediction
# -----------------------------
@app.route('/delete-prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    """Delete a single prediction by ID"""
    if 'user_id' not in session:
        flash('🔒 Please login to delete predictions!', 'error')
        return redirect(url_for('login'))
    
    try:
        # Find the prediction and ensure it belongs to the current user
        prediction = Prediction.query.filter_by(
            id=prediction_id, 
            user_id=session['user_id']
        ).first()
        
        if prediction:
            db.session.delete(prediction)
            db.session.commit()
            flash('🗑️ Prediction deleted successfully!', 'success')
        else:
            flash('❌ Prediction not found or you do not have permission to delete it.', 'error')
            
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting prediction: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

# -----------------------------
# NEW: Delete Multiple Selected Predictions
# -----------------------------
@app.route('/delete-selected', methods=['POST'])
def delete_selected():
    """Delete multiple selected predictions"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        prediction_ids = request.form.getlist('prediction_ids')
        
        if not prediction_ids:
            flash('No predictions selected.', 'error')
            return redirect(url_for('dashboard'))
        
        # Convert to integers
        prediction_ids = [int(id) for id in prediction_ids]
        
        # Delete only predictions belonging to current user
        deleted = Prediction.query.filter(
            Prediction.id.in_(prediction_ids),
            Prediction.user_id == session['user_id']
        ).delete(synchronize_session=False)
        
        db.session.commit()
        flash(f'🗑️ Successfully deleted {deleted} prediction(s).', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting predictions: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

# -----------------------------
# Debug Routes
# -----------------------------

@app.route('/debug-session')
def debug_session():
    """Debug route to check session"""
    return {
        'user_id': session.get('user_id'),
        'username': session.get('username'),
        'logged_in': 'user_id' in session,
        'openai_available': openai_available,
        'session_keys': list(session.keys())
    }

@app.route('/debug-env')
def debug_env():
    """Debug route to check environment (be careful with this in production)"""
    return {
        'openai_key_present': bool(os.getenv('OPENAI_API_KEY')),
        'openai_key_prefix': os.getenv('OPENAI_API_KEY', '')[:8] if os.getenv('OPENAI_API_KEY') else None,
        'flask_secret_present': bool(os.getenv('FLASK_SECRET_KEY')),
        'openai_available': openai_available
    }

# -----------------------------
# Error Handlers
# -----------------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template('legal.html', 
                         title="404 - Page Not Found",
                         last_updated=None,
                         content=[{
                             "title": "Oops! Page Not Found",
                             "content": "The page you're looking for doesn't exist or has been moved."
                         }]), 404

# -----------------------------
# Context Processors
# -----------------------------
@app.context_processor
def utility_processor():
    """Make current year and status available to all templates"""
    return {
        'current_year': datetime.utcnow().year,
        'openai_available': openai_available
    }

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        print("✅ Database tables created/verified")
        
        # Check if admin user exists, if not create one
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                first_name='Admin',
                last_name='User',
                username='admin',
                email='admin@spamshield.com',
                phone='+1234567890',
                country='United States',
                newsletter=False,
                password=generate_password_hash('Admin@123')
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created")
            print("   Username: admin")
            print("   Password: Admin@123")
    
    print("\n" + "="*50)
    print("🚀 SpamShield AI with OpenAI is starting...")
    print(f"🤖 OpenAI Available: {openai_available}")
    if not openai_available:
        print("✨ Using fallback explanation system (OpenAI credits needed for full AI features)")
    print("📱 Access the app at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=True)
