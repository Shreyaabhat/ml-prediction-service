"""Database table definitions (SQLAlchemy 2.0 declarative style)."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, String, Text, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("status IN ('success', 'failed')", name="ck_predictions_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Set by the database clock, the one source of truth across all API instances.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_version: Mapped[str] = mapped_column(String(32))
    input_text: Mapped[str] = mapped_column(Text)
    input_sha256: Mapped[str] = mapped_column(String(64))

    # Null when status == "failed".
    predicted_intent: Mapped[str | None] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float)
    is_out_of_scope: Mapped[bool | None] = mapped_column(Boolean)

    # Time spent computing the prediction (excludes the database write itself).
    latency_ms: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))
    error_code: Mapped[str | None] = mapped_column(String(64))