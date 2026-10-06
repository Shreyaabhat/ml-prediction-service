"""Application entry point: builds the FastAPI app and wires everything together."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_health, routes_model, routes_predict
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import request_logging_middleware
from app.db.session import create_db_engine, create_session_factory
from app.services.model_service import ModelService

logger = logging.getLogger("app.main")



@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    service = ModelService(settings.model_version)
    try:
        service.load()
        logger.info("model_loaded", extra={"fields": {"model_version": service.version}})
    except Exception:
        # Stay up but NOT ready: /health still works for diagnosis, /ready returns 503,
        # and a load balancer will not route traffic here.
        logger.exception("model_load_failed")

    # The engine connects lazily, so the app starts even if the DB is down.
    # /ready reports the DB state instead.
    engine = create_db_engine(settings)

    app.state.model_service = service
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    yield
    engine.dispose()  # close all pooled connections cleanly
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Intent Prediction Service",
        description="CLINC150 intent classification with out-of-scope detection.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware added LAST runs OUTERMOST. CORS goes last so even our generic
    # 500 responses carry the CORS headers a browser needs.
    app.middleware("http")(request_logging_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )

    register_exception_handlers(app)
    app.include_router(routes_health.router)
    app.include_router(routes_model.router)
    app.include_router(routes_predict.router)
    return app


app = create_app()