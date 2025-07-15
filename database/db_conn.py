from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

postgres_uri = os.getenv("POSTGRES_URI")
if postgres_uri is None:
    raise ValueError("POSTGRES_URI environment variable is not set")

engine = create_engine(postgres_uri)

Base = declarative_base()

class Booking(Base):
    __tablename__ = 'bookings'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    date = Column(String)
    time = Column(String)

# engine = create_engine(os.getenv("POSTGRES_URI"))
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(engine)
