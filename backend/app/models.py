import uuid
from datetime import datetime, timezone

from geoalchemy2 import Geometry
from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    task_type = Column(String(64), nullable=False, server_default="classification")
    aoi_geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=True)
    imagery_url = Column(Text, nullable=True)
    available_bands = Column(JSONB, nullable=True)   # list[str]
    enabled_indices = Column(JSONB, nullable=True)   # list[str]
    resolution_m = Column(Integer, nullable=False, server_default="10")
    model_config = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    classes = relationship("Class", back_populates="project", cascade="all, delete-orphan")
    training_samples = relationship("TrainingSample", back_populates="project", cascade="all, delete-orphan")


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    color = Column(String(7), nullable=False)          # hex, e.g. "#2d6a4f"
    display_order = Column(Integer, nullable=False, server_default="0")

    project = relationship("Project", back_populates="classes")


class TrainingSample(Base):
    __tablename__ = "training_samples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=True)
    geometry = Column(Geometry(geometry_type="POLYGON", srid=4326), nullable=False)
    label = Column(String(128), nullable=False)
    class_id = Column(Integer, nullable=True)
    image_ref = Column(Text, nullable=True)
    extra = Column("metadata", JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    project = relationship("Project", back_populates="training_samples")
    features = relationship("TrainingFeatures", back_populates="sample", cascade="all, delete-orphan")


class TrainingFeatures(Base):
    """
    Per-polygon extracted pixel features for a specific STAC scene.

    feature_data is a (n_pixels, n_features) float32 ndarray serialised
    with numpy.save() into bytes.  Deserialise with np.load(BytesIO(row.feature_data)).
    """
    __tablename__ = "training_features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_sample_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_samples.id", ondelete="CASCADE"),
        nullable=False,
    )
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    item_id = Column(Text, nullable=False)
    collection = Column(String(128), nullable=False)
    feature_names = Column(JSONB, nullable=False)   # list[str]
    n_pixels = Column(Integer, nullable=False)
    feature_data = Column(LargeBinary, nullable=False)  # numpy bytes
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    sample = relationship("TrainingSample", back_populates="features")
