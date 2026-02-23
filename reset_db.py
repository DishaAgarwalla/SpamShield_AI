# reset_db.py
from app import app, db
from werkzeug.security import generate_password_hash

with app.app_context():
    print("🗑️ Dropping all tables...")
    db.drop_all()
    
    print("🔄 Creating new tables...")
    db.create_all()
    
    # Create admin user
    print("👤 Creating admin user...")
    from app import User
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
    
    print("✅ Database reset complete!")
    print("\n📊 Admin User Created:")
    print("   Username: admin")
    print("   Password: Admin@123")
    print("   Email: admin@spamshield.com")
    
    # Optional: Create a test user
    test_user = User(
        first_name='Test',
        last_name='User',
        username='testuser',
        email='test@example.com',
        phone='+9876543210',
        country='United States',
        newsletter=True,
        password=generate_password_hash('Test@123')
    )
    db.session.add(test_user)
    db.session.commit()
    print("\n🧪 Test User Created:")
    print("   Username: testuser")
    print("   Password: Test@123")
    print("   Email: test@example.com")