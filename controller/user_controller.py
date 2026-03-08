from fastapi import HTTPException, Request
from pydantic import EmailStr,BaseModel, Field
from database.models.user_model import User
from pymongo.errors import PyMongoError
from beanie import PydanticObjectId
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

from typing import Optional

class UserResponse(BaseModel):
    id: PydanticObjectId = Field(alias="_id")
    name: str
    email: EmailStr
    profile_picture: Optional[str]
    integration_token: Optional[str]
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True



async def get_user_details(req: Request):
    """this route handle the user details fetching"""

    try:
        user_id:str = req.state.user.get("sub")
        email:EmailStr = req.state.user.get("email")

        response = await User.find_one(
            User.id == PydanticObjectId(user_id), 
            User.email == email,
            projection_model=UserResponse
        )

        print("user details response", response)
        if not response:
            raise HTTPException(status_code=404, detail="User not found")

        return {"response": response, "success": True}
    
    except PyMongoError as e:
        logger.exception("DB error during fetching user details")
        raise HTTPException(status_code=503, detail="Database unavailable")