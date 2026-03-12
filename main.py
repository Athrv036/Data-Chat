from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# IMPORT ALL ROUTERS
from app.routers import datasets, chat, ml   # <-- ADDED ml HERE

app = FastAPI(title="DataChat Backend")

# ==============================
# CORS SETTINGS
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow frontend to call backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# ROUTERS
# ==============================
app.include_router(datasets.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(ml.router, prefix="/api")   # <-- ADDED THIS LINE

# ==============================
# HEALTH CHECK
# ==============================
@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Backend running successfully"}
