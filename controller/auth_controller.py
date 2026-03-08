from __future__ import annotations

from typing import TYPE_CHECKING

from database.models.user_model import User
from fastapi import HTTPException, status
from controller.auth.password_utils import verify_password, hash_password
from controller.auth.jwt_create_token import create_access_token
from pymongo.errors import PyMongoError
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from server import CreateUserRequest, VerifyUserRequest


# rehister controller for registering the new user to the database
async def register_user(req: CreateUserRequest):
    # check if user with the same email already exists
    try:
        existing = await User.find_one(User.email == req.email)
        if existing:
            raise ValueError("User with this email already exists. Try Signin instead")
        
        hashed_password = hash_password(req.password)

        user = User(
            name=req.name, 
            email=req.email, 
            password_hash=hashed_password, 
            profile_picture=req.profile_picture   
        )

        await user.insert()

        jwt_token = create_access_token({
            "sub": str(user.id),
            "email": user.email
        })

        return {"sub": str(user.id), "name": user.name, "email": user.email, "jwt_token": jwt_token}
    
    except PyMongoError as e:
        logger.exception("DB error during register")
        raise HTTPException(status_code=503, detail="Database unavailable")
    

# verify controller for verifying the user credentials and return the jwt token
async def verify_user(req: VerifyUserRequest):
    # check if user with the provided email exists
    try:
        user = await User.find_one(User.email == req.email)
        if not user:
            raise ValueError("Invalid email or password")

        # check if the provided password matches the user's password
        if not verify_password(req.password, user.password_hash):
            raise ValueError("Invalid email or password")

        jwt_token = create_access_token({
            "sub": str(user.id),
            "email": user.email
        })

        return {"sub": str(user.id), "name": user.name, "email": user.email, "jwt_token": jwt_token}
    except PyMongoError as e:
        logger.exception("DB error during verify")
        raise HTTPException(status_code=503, detail="Database unavailable")