"""Queue, persistence and history behaviour of jobs.py."""
import base64
import json

import pytest

import config
import jobs
import pipeline
from conftest import wait_for, write_job_json

IMAGE = base64.b64encode(b"pretend this is a PNG").decode()


def test_submit_returns_a_queued_record_carrying_its_params():
    job = jobs.submit("image_to_3d", {"seed": 7}, IMAGE)

    assert job["status"] == jobs.QUEUED
    assert job["params"] == {"seed": 7}
    assert job["result"] is None and job["error"] is None
    assert job["started_at"] is None and job["finished_at"] is None
    assert len(job["id"]) == 12


def test_submit_enqueues_the_job_id():
    job = jobs.submit("image_to_3d", {}, IMAGE)

    assert jobs._queue.qsize() == 1
    assert jobs._queue.get_nowait() == job["id"]


def test_the_input_image_is_never_written_to_job_json():
    job = jobs.submit("image_to_3d", {}, IMAGE)

    raw = (config.OUT_DIR / job["id"] / "job.json").read_text()
    assert IMAGE not in raw
    assert "image" not in json.loads(raw)
    assert "image" not in job


def test_the_input_image_is_never_returned_by_get_or_listing():
    job = jobs.submit("image_to_3d", {}, IMAGE)

    assert IMAGE not in json.dumps(jobs.get(job["id"]))
    assert IMAGE not in json.dumps(jobs.listing())


def test_get_reads_job_json_when_the_job_is_no_longer_in_memory():
    job = jobs.submit("image_to_3d", {"seed": 3}, IMAGE)
    jobs._run_one(job["id"])
    expected = jobs.get(job["id"])

    jobs._jobs.clear()

    assert jobs.get(job["id"]) == expected


def test_get_returns_none_for_an_unknown_job():
    assert jobs.get("deadbeefcafe") is None


def test_get_returns_none_when_job_json_is_corrupt():
    d = config.OUT_DIR / "brokenjob"
    d.mkdir()
    (d / "job.json").write_text("{not json")

    assert jobs.get("brokenjob") is None


def test_listing_returns_newest_first():
    ids = [jobs.submit("image_to_3d", {}, IMAGE)["id"] for _ in range(3)]

    assert [j["id"] for j in jobs.listing()] == ids[::-1]


def test_listing_limit_keeps_the_newest_jobs():
    ids = [jobs.submit("image_to_3d", {}, IMAGE)["id"] for _ in range(5)]

    assert [j["id"] for j in jobs.listing(limit=2)] == ids[-2:][::-1]


def test_listing_with_a_non_positive_limit_returns_nothing():
    jobs.submit("image_to_3d", {}, IMAGE)

    assert jobs.listing(limit=0) == []
    assert jobs.listing(limit=-1) == []


def test_listing_returns_copies_that_cannot_mutate_the_registry():
    job = jobs.submit("image_to_3d", {}, IMAGE)

    jobs.listing()[0]["status"] = "tampered"

    assert jobs.get(job["id"])["status"] == jobs.QUEUED


def test_stats_reports_queue_depth_and_running_ids():
    queued = jobs.submit("image_to_3d", {}, IMAGE)
    running = jobs.submit("image_to_3d", {}, IMAGE)
    jobs._set(running["id"], status=jobs.RUNNING)

    assert jobs.stats() == {"queue_depth": 2, "running": [running["id"]]}
    assert queued["id"] not in jobs.stats()["running"]


def test_run_one_records_the_pipeline_result_and_marks_the_job_done():
    job = jobs.submit("image_to_3d", {"seed": 1}, IMAGE)

    jobs._run_one(job["id"])

    done = jobs.get(job["id"])
    assert done["status"] == jobs.DONE
    assert done["error"] is None
    assert done["started_at"] and done["finished_at"]
    assert done["result"]["faces"] == 12
    assert (config.OUT_DIR / job["id"] / "mesh.glb").exists()


def test_run_one_forgets_the_image_once_the_job_ends():
    job = jobs.submit("image_to_3d", {}, IMAGE)

    jobs._run_one(job["id"])

    assert job["id"] not in jobs._images


def test_run_one_records_the_exception_type_and_message_on_failure(monkeypatch):
    def boom(image_b64, out_dir, params):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "generate_shape", boom)
    job = jobs.submit("image_to_3d", {}, IMAGE)

    jobs._run_one(job["id"])

    failed = jobs.get(job["id"])
    assert failed["status"] == jobs.ERROR
    assert failed["error"] == "RuntimeError: CUDA out of memory"
    assert failed["finished_at"] is not None
    assert job["id"] not in jobs._images


def test_run_one_defaults_to_the_configured_generator():
    job = jobs.submit("image_to_3d", {}, IMAGE)

    jobs._run_one(job["id"])

    assert jobs.get(job["id"])["result"]["generator"] == config.DEFAULT_GENERATOR


def test_run_one_dispatches_to_the_generator_the_params_name():
    job = jobs.submit("image_to_3d", {"generator": "trellis2"}, IMAGE)

    jobs._run_one(job["id"])

    assert jobs.get(job["id"])["result"]["generator"] == "trellis2"


def test_run_one_rejects_an_unknown_generator():
    job = jobs.submit("image_to_3d", {"generator": "trellis3"}, IMAGE)

    jobs._run_one(job["id"])

    assert jobs.get(job["id"])["error"] == "ValueError: unknown generator: trellis3"


def test_both_generators_expose_the_same_interface():
    """The dispatch table is the whole abstraction; there is no base class to
    make this true, so it is asserted instead."""
    for module in jobs.GENERATORS.values():
        assert callable(module.generate_shape)
        assert callable(module.unload)
        assert module.model_loaded() is False


def test_run_one_rejects_an_unknown_job_type():
    job = jobs.submit("text_to_3d", {}, IMAGE)

    jobs._run_one(job["id"])

    assert jobs.get(job["id"])["error"] == "ValueError: unknown job type: text_to_3d"


def test_run_one_rejects_a_job_with_no_image():
    job = jobs.submit("image_to_3d", {}, None)

    jobs._run_one(job["id"])

    assert jobs.get(job["id"])["error"] == "ValueError: image_to_3d requires an image"


def test_run_one_ignores_an_unknown_job_id():
    jobs._run_one("nosuchjobid12")

    assert jobs.listing() == []


def test_history_evicts_the_oldest_finished_job(monkeypatch):
    monkeypatch.setattr(config, "MAX_JOB_HISTORY", 3)
    ids = [jobs.submit("image_to_3d", {}, IMAGE)["id"] for _ in range(3)]
    for job_id in ids:
        jobs._set(job_id, status=jobs.DONE)

    newest = jobs.submit("image_to_3d", {}, IMAGE)["id"]

    assert list(jobs._jobs) == ids[1:] + [newest]
    assert ids[0] not in jobs._images


def test_an_evicted_job_is_still_readable_from_disk(monkeypatch):
    monkeypatch.setattr(config, "MAX_JOB_HISTORY", 1)
    evicted = jobs.submit("image_to_3d", {}, IMAGE)["id"]
    jobs._set(evicted, status=jobs.DONE)

    jobs.submit("image_to_3d", {}, IMAGE)

    assert evicted not in jobs._jobs
    assert jobs.get(evicted)["status"] == jobs.DONE


def test_history_never_evicts_queued_or_running_work(monkeypatch):
    monkeypatch.setattr(config, "MAX_JOB_HISTORY", 2)
    still_queued = jobs.submit("image_to_3d", {}, IMAGE)["id"]
    running = jobs.submit("image_to_3d", {}, IMAGE)["id"]
    jobs._set(running, status=jobs.RUNNING)

    for _ in range(3):
        jobs.submit("image_to_3d", {}, IMAGE)

    # The cap is deliberately breached rather than dropping work in flight; the
    # unfinished jobs also keep their images, which the worker still needs.
    assert still_queued in jobs._jobs and running in jobs._jobs
    assert len(jobs._jobs) > config.MAX_JOB_HISTORY
    assert jobs._images[still_queued] == IMAGE


def test_eviction_resumes_once_the_oldest_job_finishes(monkeypatch):
    monkeypatch.setattr(config, "MAX_JOB_HISTORY", 2)
    first = jobs.submit("image_to_3d", {}, IMAGE)["id"]
    for _ in range(2):
        jobs.submit("image_to_3d", {}, IMAGE)
    assert first in jobs._jobs

    jobs._set(first, status=jobs.DONE)
    jobs.submit("image_to_3d", {}, IMAGE)

    assert first not in jobs._jobs


def test_rehydrate_loads_finished_jobs_oldest_first(out_dir):
    write_job_json(out_dir, "aaaaaaaaaaaa", jobs.DONE, created_at=100.0)
    write_job_json(out_dir, "bbbbbbbbbbbb", jobs.DONE, created_at=200.0)

    jobs.rehydrate()

    assert list(jobs._jobs) == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    assert [j["id"] for j in jobs.listing()] == ["bbbbbbbbbbbb", "aaaaaaaaaaaa"]


@pytest.mark.parametrize("status", [jobs.QUEUED, jobs.RUNNING])
def test_rehydrate_fails_jobs_that_were_in_flight_at_shutdown(out_dir, status):
    write_job_json(out_dir, "inflight1234", status)

    jobs.rehydrate()

    job = jobs._jobs["inflight1234"]
    assert job["status"] == jobs.ERROR
    assert job["error"] == "server restarted while this job was in flight"


@pytest.mark.parametrize("status", [jobs.QUEUED, jobs.RUNNING])
def test_rehydrate_writes_the_failure_back_to_disk(out_dir, status):
    """Correcting only the in-memory copy leaves job.json saying "running", and
    get() reads that stale status back once the job is evicted from memory."""
    write_job_json(out_dir, "inflight1234", status)

    jobs.rehydrate()
    jobs._jobs.clear()

    assert jobs.get("inflight1234")["status"] == jobs.ERROR


def test_rehydrate_leaves_errored_jobs_alone(out_dir):
    write_job_json(out_dir, "failedjob123", jobs.ERROR, error="RuntimeError: nope")

    jobs.rehydrate()

    assert jobs._jobs["failedjob123"]["error"] == "RuntimeError: nope"


def test_rehydrate_skips_unreadable_job_json(out_dir):
    write_job_json(out_dir, "goodjob12345", jobs.DONE)
    (out_dir / "badjob123456").mkdir()
    (out_dir / "badjob123456" / "job.json").write_text("}{")

    jobs.rehydrate()

    assert list(jobs._jobs) == ["goodjob12345"]


def test_rehydrate_keeps_only_the_newest_max_history_jobs(out_dir, monkeypatch):
    monkeypatch.setattr(config, "MAX_JOB_HISTORY", 2)
    for i in range(5):
        write_job_json(out_dir, f"job{i:09d}", jobs.DONE, created_at=float(i))

    jobs.rehydrate()

    assert list(jobs._jobs) == ["job000000003", "job000000004"]


def test_rehydrate_does_not_overwrite_a_job_already_in_memory(out_dir):
    live = jobs.submit("image_to_3d", {}, IMAGE)
    write_job_json(out_dir, live["id"], jobs.ERROR, error="stale")

    jobs.rehydrate()

    assert jobs._jobs[live["id"]]["status"] == jobs.QUEUED


def test_rehydrate_is_a_no_op_when_the_output_directory_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUT_DIR", tmp_path / "never-created")

    jobs.rehydrate()

    assert jobs._jobs == {}


def test_the_worker_thread_runs_a_queued_job_to_completion():
    jobs.start_worker()
    job = jobs.submit("image_to_3d", {}, IMAGE)

    done = wait_for(job["id"], jobs.DONE)

    assert done["result"]["mesh_path"].endswith("mesh.glb")


def test_start_worker_does_not_start_a_second_thread():
    jobs.start_worker()
    first = jobs._worker

    jobs.start_worker()

    assert jobs._worker is first and first.is_alive()
