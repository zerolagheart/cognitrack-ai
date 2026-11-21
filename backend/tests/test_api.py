"""API, database and error-handling tests against a temporary SQLite database."""

from __future__ import annotations

import io

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(cfg, pipeline_outputs, workspace, tmp_path_factory):
    import os

    # CI also runs this suite against PostgreSQL via COGNITRACK_TEST_DATABASE_URL.
    os.environ["COGNITRACK_DATABASE_URL"] = os.environ.get("COGNITRACK_TEST_DATABASE_URL") or f"sqlite:///{(workspace / 'api.db').as_posix()}"
    os.environ["COGNITRACK_UPLOAD_DIR"] = str(tmp_path_factory.mktemp("uploads"))
    os.environ["COGNITRACK_ENVIRONMENT"] = "test"
    for k in ("VIEWER", "RESEARCHER", "ADMIN"):
        os.environ.pop(f"COGNITRACK_{k}_API_KEYS", None)
    from backend.app.core.config import get_settings
    from backend.app.database.session import get_engine, get_sessionmaker
    from backend.app.services.predictor import reset_predictor

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    reset_predictor()
    from backend.app.main import create_app
    from backend.app.services.sync import publish_all

    counts = publish_all()
    assert counts["recordings"] == len(pipeline_outputs["manifest"])
    with TestClient(create_app()) as c:
        yield c


def _wav_bytes(seconds: float = 6.0) -> bytes:
    from conftest import tone_bursts

    buf = io.BytesIO()
    sig = tone_bursts(bursts=int(seconds / 0.9))
    sf.write(buf, sig, 16000, format="WAV")
    return buf.getvalue()


def test_health(client):
    r = client.get("/api/v1/health").json()
    assert r["status"] == "ok" and r["database"] and r["model_loaded"]
    assert "not a medical device" in r["disclaimer"].lower()


def test_overview_counts(client, pipeline_outputs):
    r = client.get("/api/v1/overview").json()
    assert r["counts"]["participants"] == pipeline_outputs["manifest"]["participant_id"].nunique()
    assert r["synthetic_data"] is True and r["serving_model"] is not None


def test_participant_endpoints(client, pipeline_outputs):
    pid = pipeline_outputs["manifest"]["participant_id"].iloc[0]
    lst = client.get("/api/v1/participants", params={"search": pid}).json()
    assert lst["total"] >= 1
    detail = client.get(f"/api/v1/participants/{pid}").json()
    assert detail["sessions"] and detail["is_synthetic"]
    trends = client.get(f"/api/v1/participants/{pid}/features", params={"modality": "acoustic"}).json()
    assert trends["features"] and trends["sessions"]
    lon = client.get(f"/api/v1/participants/{pid}/longitudinal").json()
    assert lon["status"] in {"analysed", "insufficient_data"} and "interpretation" in lon
    assert client.get("/api/v1/participants/NOPE-123").status_code == 404


def test_recording_detail_waveform_predict_explain(client, pipeline_outputs):
    df = pipeline_outputs["dataset"]
    key = df.loc[df["quality_usable"].astype(bool), "recording_id"].iloc[0]
    rec = client.get(f"/api/v1/recordings/{key}").json()
    assert rec["features"] and rec["quality_usable"]
    wf = client.get(f"/api/v1/recordings/{key}/waveform").json()
    assert len(wf["min"]) == len(wf["max"]) > 100
    pred = client.post(f"/api/v1/recordings/{key}/predict").json()
    assert 0 <= pred["probability"] <= 1 and pred["origin"] == "api"
    expl = client.get(f"/api/v1/recordings/{key}/explanation").json()
    assert expl["explanation"]["method"]


def test_reports_and_models(client):
    for rtype in ("model_comparison", "explainability", "fairness", "drift", "longitudinal", "model_card", "data_quality"):
        r = client.get(f"/api/v1/reports/{rtype}")
        assert r.status_code == 200, rtype
        assert r.json()["is_synthetic"]
    assert client.get("/api/v1/reports/unknown").status_code == 404
    models = client.get("/api/v1/models").json()
    assert sum(m["is_serving"] for m in models) == 1
    assert client.get("/api/v1/experiments").json()
    assert client.get("/api/v1/longitudinal/cohort").json()["cohort_summary"]["n_participants"] > 0


def test_upload_processes_and_deletes_raw_audio(client):
    files = {"file": ("session.wav", _wav_bytes(), "audio/wav")}
    data = {"participant": "STUDY-0001", "consent_confirmed": "true", "transcript": "the boy is taking a cookie. the girl is laughing."}
    r = client.post("/api/v1/recordings/upload", files=files, data=data)
    assert r.status_code == 202, r.text
    body = r.json()
    job = client.get(f"/api/v1/monitoring/jobs/{body['job']['id']}").json()
    assert job["status"] == "succeeded", job
    rec = client.get(f"/api/v1/recordings/{body['recording_key']}").json()
    assert rec["raw_audio_retained"] is False and rec["source"] == "upload"
    assert rec["latest_prediction"] is not None
    assert client.get(f"/api/v1/recordings/{body['recording_key']}/waveform").status_code == 200  # stored envelope
    preds = client.get("/api/v1/participants/STUDY-0001/predictions").json()
    assert len(preds) == 1


@pytest.mark.parametrize("files, data, status", [
    ({"file": ("a.wav", b"", "audio/wav")}, {"participant": "STUDY-1", "consent_confirmed": "true"}, 422),
    ({"file": ("a.txt", b"hello", "text/plain")}, {"participant": "STUDY-1", "consent_confirmed": "true"}, 422),
    ({"file": ("a.wav", b"RIFFxxxxWAVEjunk", "audio/wav")}, {"participant": "STUDY-1", "consent_confirmed": "true"}, 422),
    (None, {"participant": "STUDY-1", "consent_confirmed": "false"}, 422),
    (None, {"participant": "john.doe@example.com", "consent_confirmed": "true"}, 422),
])
def test_upload_validation(client, files, data, status):
    files = files or {"file": ("a.wav", _wav_bytes(3), "audio/wav")}
    r = client.post("/api/v1/recordings/upload", files=files, data=data)
    assert r.status_code == status
    assert "error" in r.json()


def test_upload_size_limit(client, monkeypatch):
    from backend.app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0.01)
    r = client.post("/api/v1/recordings/upload", files={"file": ("big.wav", _wav_bytes(5), "audio/wav")},
                    data={"participant": "STUDY-2", "consent_confirmed": "true"})
    assert r.status_code == 413


def test_feature_prediction_schema(client):
    r = client.post("/api/v1/predict/features", json={"features": {}})
    assert r.status_code == 422
    r = client.post("/api/v1/predict/features", json={"features": {"pause_rate_per_min": 12.0}})
    assert r.status_code == 200 and r.json()["uncertainty"]["abstain"]


def test_auth_roles(client, monkeypatch):
    from backend.app.core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "viewer_api_keys", "view-key")
    monkeypatch.setattr(s, "researcher_api_keys", "research-key")
    assert client.get("/api/v1/overview").status_code == 401
    assert client.get("/api/v1/overview", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/overview", headers={"X-API-Key": "view-key"}).status_code == 200
    assert client.get("/api/v1/participants", headers={"X-API-Key": "view-key"}).status_code == 403
    assert client.get("/api/v1/participants", headers={"X-API-Key": "research-key"}).status_code == 200
    assert client.post("/api/v1/admin/publish", headers={"X-API-Key": "research-key"}).status_code == 403


def test_database_constraints(client):
    from backend.app.database.session import get_sessionmaker
    from backend.app.models import ModelVersion, Prediction, Recording

    with get_sessionmaker()() as db:
        rec = db.query(Recording).first()
        mv = db.query(ModelVersion).first()
        db.add(Prediction(recording_id=rec.id, model_version_id=mv.id, probability=1.5, model_label="x"))
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        db.add(Recording(recording_key=rec.recording_key, session_id=rec.session_id, participant_id=rec.participant_id, dataset_source="x"))
        with pytest.raises(IntegrityError):
            db.commit()


def test_audit_events_recorded(client):
    from backend.app.database.session import get_sessionmaker
    from backend.app.models import AuditEvent

    with get_sessionmaker()() as db:
        assert db.query(AuditEvent).filter(AuditEvent.resource == "participant").count() >= 1


def test_republish_keeps_exactly_one_serving_model(client):
    """Regression: publishing twice must not clear the serving flag."""
    from backend.app.services.sync import publish_all

    publish_all()
    models = client.get("/api/v1/models").json()
    assert sum(m["is_serving"] for m in models) == 1
    assert client.get("/api/v1/overview").json()["serving_model"] is not None
