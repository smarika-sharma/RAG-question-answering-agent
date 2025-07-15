from fastapi import FastAPI
from api import routes_upload, routes_chat

app= FastAPI()

# first api route
app.include_router(routes_upload.router)

# second api route
app.include_router(routes_chat.router)

