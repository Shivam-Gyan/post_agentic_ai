import bcrypt


def check_correct_password_format(password: str):
    error_messages = []
    if len(password) < 8:
        error_messages.append("Password must be at least 8 characters long")
    if not any(c.isupper() for c in password):
        error_messages.append("Password must contain at least one uppercase letter")
    if not any(c.islower() for c in password):
        error_messages.append("Password must contain at least one lowercase letter")
    if not any(c.isdigit() for c in password):
        error_messages.append("Password must contain at least one digit")
    if not any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?/" for c in password):
        error_messages.append("Password must contain at least one special character")

    if error_messages:
        raise ValueError(",".join(error_messages))

def hash_password(password: str) -> str:
    check_correct_password_format(password) # Ensure password meets complexity requirements
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode(), salt)
    return hashed.decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())