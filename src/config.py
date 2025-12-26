import os
from dotenv import load_dotenv

load_dotenv()

OSRM_HOST = os.getenv("OSRM_HOST")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RANDOM_SEED = int(os.getenv("RANDOM_SEED"))
DC_CONFIG = {'store_id': 'dc', 'longitude': float(os.getenv("DC_LONGITUDE")), 'latitude': float(os.getenv("DC_LATITUDE"))}
