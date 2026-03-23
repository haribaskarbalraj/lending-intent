import logging

from fastapi import FastAPI

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

from src.health.router import router as health_router
from src.intent.router import router as intent_router

app = FastAPI(
    title="Lending Intent Engine",
    description="Analyses customer spending patterns to classify lending intent using AWS Bedrock",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(intent_router)
