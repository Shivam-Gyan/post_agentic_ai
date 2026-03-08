from jose import jwt, JWTError
from fastapi import HTTPException, status
from controller.auth.jwt_create_token import SECRET_KEY, ALGORITHM


def verify_token(token: str):

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # here sub is the user_id that we encoded in the token
        user_id: str = payload.get("sub") #type: ignore[union-attr]

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

        return payload

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token verification failed"
        )