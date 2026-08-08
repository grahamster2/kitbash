"""HTTP surface, driven through FastAPI's TestClient with generation stubbed."""
import json
from pathlib import Path

import pytest

import jobs
from conftest import PNG_B64, PNG_BYTES


def test_health_answers_on_a_machine_with_no_cuda(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["gpu"] is None
    assert body["queue_depth"] == 0 and body["running"] == []


def test_post_jobs_queues_the_work_and_returns_its_id(client):
    body = client.post("/jobs", json={"image_b64": PNG_B64}).json()

    assert body["status"] == jobs.QUEUED
    assert client.get("/health").json()["queue_depth"] == 1
    assert client.get(f"/jobs/{body['id']}").json()["id"] == body["id"]


def test_post_jobs_forwards_only_the_params_the_caller_set(client):
    body = client.post(
        "/jobs", json={"image_b64": PNG_B64, "seed": 5, "target_faces": 8000}
    ).json()

    assert body["params"] == {"seed": 5, "target_faces": 8000}


def test_post_jobs_defaults_to_no_params(client):
    body = client.post("/jobs", json={"image_b64": PNG_B64}).json()

    assert body["params"] == {}


def test_post_jobs_takes_a_generator_and_a_textured_flag(client):
    body = client.post(
        "/jobs", json={"image_b64": PNG_B64, "generator": "trellis2", "textured": False}
    ).json()

    assert body["params"] == {"generator": "trellis2", "textured": False}


def test_post_jobs_rejects_an_unknown_generator(client):
    """Caught at the call that made the typo rather than by the worker two
    minutes into a queue."""
    response = client.post(
        "/jobs", json={"image_b64": PNG_B64, "generator": "trellis3"}
    )

    assert response.status_code == 400
    assert "trellis3" in response.json()["detail"]


def test_a_job_records_which_generator_built_it(client, finished_job):
    for name in ("hunyuan3d", "trellis2"):
        job_id = finished_job(generator=name)

        assert client.get(f"/jobs/{job_id}").json()["result"]["generator"] == name


def test_generators_lists_both_with_their_measured_costs(client):
    body = client.get("/generators").json()

    assert body["default"] == "hunyuan3d"
    by_name = {g["name"]: g for g in body["generators"]}
    assert set(by_name) == {"hunyuan3d", "trellis2"}
    assert by_name["hunyuan3d"]["textures"] is False
    assert by_name["trellis2"]["textures"] is True
    # The recommended-but-unusable settings must not be advertised as defaults.
    assert by_name["trellis2"]["defaults"]["pipeline_type"] == "512"
    assert by_name["trellis2"]["defaults"]["texture_size"] == 2048
    assert all("peak_vram_gib" in g["characteristics"] for g in body["generators"])


def test_generators_reports_trellis_as_unavailable_without_its_install(client):
    """Hunyuan3D is in-process, so it ships with the server; TRELLIS 2 is a
    separate 28 GB install that a given box may simply not have."""
    body = client.get("/generators").json()
    trellis2 = next(g for g in body["generators"] if g["name"] == "trellis2")

    assert trellis2["available"] is False
    assert trellis2["missing"]
    assert trellis2["in_process"] is False


def test_health_reports_residency_per_generator(client):
    body = client.get("/health").json()

    assert body["generators_loaded"] == {"hunyuan3d": False, "trellis2": False}


def test_post_jobs_requires_an_image(client):
    response = client.post("/jobs", json={"part_name": "hull"})

    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_post_jobs_rejects_both_an_image_and_an_image_id(client):
    response = client.post("/jobs", json={"image_b64": PNG_B64, "image_id": "abc"})

    assert response.status_code == 400


def test_post_jobs_accepts_a_stored_image_id(client, out_dir):
    import imagegen

    image_id, _ = imagegen.store(PNG_BYTES, remove_background=False)

    assert client.post("/jobs", json={"image_id": image_id}).status_code == 200


def test_post_jobs_reports_an_unknown_image_id(client):
    response = client.post("/jobs", json={"image_id": "missing12345"})

    assert response.status_code == 404


def test_post_jobs_never_echoes_the_image_back(client):
    response = client.post("/jobs", json={"image_b64": PNG_B64})

    assert PNG_B64 not in response.text
    assert "image_b64" not in response.json()


def test_get_jobs_lists_newest_first(client):
    first = client.post("/jobs", json={"image_b64": PNG_B64}).json()["id"]
    second = client.post("/jobs", json={"image_b64": PNG_B64}).json()["id"]

    listed = client.get("/jobs").json()["jobs"]

    assert [j["id"] for j in listed] == [second, first]


def test_get_jobs_honours_the_limit(client):
    for _ in range(3):
        client.post("/jobs", json={"image_b64": PNG_B64})

    assert len(client.get("/jobs", params={"limit": 2}).json()["jobs"]) == 2
    assert client.get("/jobs", params={"limit": 0}).json()["jobs"] == []


def test_get_job_404s_for_an_unknown_id(client):
    response = client.get("/jobs/nosuchjob123")

    assert response.status_code == 404
    assert "nosuchjob123" in response.json()["detail"]


def test_get_job_returns_the_finished_record(client, finished_job):
    job_id = finished_job()

    body = client.get(f"/jobs/{job_id}").json()

    assert body["status"] == jobs.DONE
    assert body["result"]["faces"] == 12


def test_get_mesh_serves_the_glb(client, finished_job):
    job_id = finished_job()

    response = client.get(f"/jobs/{job_id}/mesh")

    assert response.status_code == 200
    assert response.headers["content-type"] == "model/gltf-binary"
    assert response.content[:4] == b"glTF"


def test_get_mesh_409s_while_the_job_is_unfinished(client):
    job_id = client.post("/jobs", json={"image_b64": PNG_B64}).json()["id"]

    response = client.get(f"/jobs/{job_id}/mesh")

    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_get_mesh_404s_for_an_unknown_job(client):
    assert client.get("/jobs/nosuchjob123/mesh").status_code == 404


def test_describe_reports_the_part_size(client, finished_job):
    job_id = finished_job()

    body = client.get(f"/jobs/{job_id}/describe").json()

    assert body["size"] == [1.0, 2.0, 3.0]
    assert body["faces"] == 12


def test_describe_404s_and_409s_like_the_mesh_endpoint(client):
    queued = client.post("/jobs", json={"image_b64": PNG_B64}).json()["id"]

    assert client.get("/jobs/nosuchjob123/describe").status_code == 404
    assert client.get(f"/jobs/{queued}/describe").status_code == 409


def test_assemble_composes_finished_parts_into_one_scene(client, finished_job, out_dir):
    hull, wing = finished_job("hull"), finished_job("wing")

    body = client.post(
        "/assemble",
        json={
            "scene_name": "plane",
            "parts": [
                {"job_id": hull, "name": "hull"},
                {"job_id": wing, "name": "wing", "position": [10, 0, 0]},
            ],
        },
    ).json()

    assert body["part_count"] == 2
    assert [p["name"] for p in body["parts"]] == ["hull", "wing"]
    assert body["size"] == [11.0, 2.0, 3.0]
    assert body["scene_path"] == str(
        out_dir / "scenes" / body["scene_id"] / "plane.glb"
    )
    assert Path(body["scene_path"]).is_file()


def test_assemble_defaults_the_scene_filename(client, finished_job):
    job_id = finished_job()

    body = client.post(
        "/assemble", json={"parts": [{"job_id": job_id, "name": "hull"}]}
    ).json()

    assert body["scene_path"].endswith("scene.glb")


def test_assemble_rejects_an_empty_part_list(client):
    response = client.post("/assemble", json={"parts": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "no parts given"


def test_assemble_404s_when_a_part_job_does_not_exist(client):
    response = client.post(
        "/assemble", json={"parts": [{"job_id": "nosuchjob123", "name": "hull"}]}
    )

    assert response.status_code == 404


def test_assemble_409s_when_a_part_job_is_unfinished(client):
    queued = client.post("/jobs", json={"image_b64": PNG_B64}).json()["id"]

    response = client.post(
        "/assemble", json={"parts": [{"job_id": queued, "name": "hull"}]}
    )

    assert response.status_code == 409


def test_assemble_404s_when_use_raw_has_no_dense_original(client, finished_job):
    job_id = finished_job()

    response = client.post(
        "/assemble",
        json={"parts": [{"job_id": job_id, "name": "hull", "use_raw": True}]},
    )

    assert response.status_code == 404
    assert "mesh_raw.glb" in response.json()["detail"]


def test_assemble_uses_the_dense_original_when_asked(client, finished_job, out_dir):
    job_id = finished_job()
    mesh = out_dir / job_id / "mesh.glb"
    (out_dir / job_id / "mesh_raw.glb").write_bytes(mesh.read_bytes())

    body = client.post(
        "/assemble",
        json={"parts": [{"job_id": job_id, "name": "hull", "use_raw": True}]},
    ).json()

    assert body["parts"][0]["source"].endswith("mesh_raw.glb")


def test_assemble_400s_when_a_part_mesh_has_gone_missing(client, finished_job, out_dir):
    job_id = finished_job()
    (out_dir / job_id / "mesh.glb").unlink()

    response = client.post(
        "/assemble", json={"parts": [{"job_id": job_id, "name": "hull"}]}
    )

    assert response.status_code == 400
    assert "part mesh missing" in response.json()["detail"]


def test_get_scene_serves_the_assembled_glb(client, finished_job):
    job_id = finished_job()
    scene_id = client.post(
        "/assemble", json={"parts": [{"job_id": job_id, "name": "hull"}]}
    ).json()["scene_id"]

    response = client.get(f"/scenes/{scene_id}/mesh")

    assert response.status_code == 200
    assert response.content[:4] == b"glTF"


def test_get_scene_404s_for_an_unknown_scene(client):
    assert client.get("/scenes/nosuchscene/mesh").status_code == 404


def test_export_requires_exactly_one_of_job_id_or_scene_id(client, finished_job):
    job_id = finished_job()

    assert client.post("/export", json={"target": "roblox"}).status_code == 400
    both = client.post(
        "/export", json={"job_id": job_id, "scene_id": "abc", "target": "roblox"}
    )
    assert both.status_code == 400
    assert "exactly one" in both.json()["detail"]


def test_export_writes_the_target_files_under_the_job_directory(
    client, finished_job, out_dir
):
    job_id = finished_job()

    body = client.post("/export", json={"job_id": job_id, "target": "roblox"}).json()

    assert body["primary"] == str(out_dir / job_id / "export" / "roblox" / "mesh.glb")
    assert Path(body["primary"]).is_file()
    assert body["pivot"] == "base-centered"


def test_export_applies_height_studs(client, finished_job):
    job_id = finished_job()

    body = client.post(
        "/export", json={"job_id": job_id, "target": "roblox", "height_studs": 12}
    ).json()

    assert body["size"][1] == 12.0


def test_export_rejects_an_unknown_target(client, finished_job):
    job_id = finished_job()

    response = client.post("/export", json={"job_id": job_id, "target": "unity"})

    assert response.status_code == 400
    assert "unknown target" in response.json()["detail"]


def test_export_404s_and_409s_on_the_job_it_is_given(client):
    queued = client.post("/jobs", json={"image_b64": PNG_B64}).json()["id"]

    assert client.post("/export", json={"job_id": "nosuchjob123"}).status_code == 404
    assert client.post("/export", json={"job_id": queued}).status_code == 409


def test_export_404s_for_an_unknown_scene(client):
    assert client.post("/export", json={"scene_id": "nosuchscene"}).status_code == 404


def test_export_writes_a_scene_export_beside_the_scene(client, finished_job, out_dir):
    job_id = finished_job()
    scene_id = client.post(
        "/assemble", json={"parts": [{"job_id": job_id, "name": "hull"}]}
    ).json()["scene_id"]

    body = client.post("/export", json={"scene_id": scene_id, "target": "dcc"}).json()

    assert body["primary"].startswith(str(out_dir / "scenes" / scene_id / "export"))
    assert Path(body["primary"]).is_file()


def test_export_file_serves_a_file_inside_the_output_directory(
    client, finished_job, out_dir
):
    job_id = finished_job()
    primary = client.post(
        "/export", json={"job_id": job_id, "target": "roblox"}
    ).json()["primary"]

    response = client.get("/export/file", params={"path": primary})

    assert response.status_code == 200
    assert response.content == Path(primary).read_bytes()


def test_export_file_404s_for_a_missing_file_inside_the_output_directory(
    client, out_dir
):
    response = client.get("/export/file", params={"path": str(out_dir / "nope.glb")})

    assert response.status_code == 404


def test_export_file_404s_for_a_directory(client, out_dir):
    response = client.get("/export/file", params={"path": str(out_dir)})

    assert response.status_code == 404


@pytest.mark.parametrize("path", ["/etc/passwd", "/etc/hostname"])
def test_export_file_refuses_an_absolute_path_outside_the_output_directory(
    client, path
):
    response = client.get("/export/file", params={"path": path})

    assert response.status_code == 403
    assert response.json()["detail"] == "path is outside the output directory"


def test_export_file_refuses_a_path_that_climbs_out_of_the_output_directory(
    client, out_dir, tmp_path
):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")

    response = client.get(
        "/export/file", params={"path": str(out_dir / ".." / "secret.txt")}
    )

    assert response.status_code == 403


def test_export_file_refuses_a_symlink_that_points_outside(client, out_dir, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("private")
    link = out_dir / "innocent.glb"
    link.symlink_to(secret)

    response = client.get("/export/file", params={"path": str(link)})

    assert response.status_code == 403


def test_export_file_refuses_a_sibling_directory_sharing_the_prefix(client, tmp_path):
    evil = tmp_path / "out-evil"
    evil.mkdir()
    (evil / "loot.glb").write_text("loot")

    response = client.get("/export/file", params={"path": str(evil / "loot.glb")})

    assert response.status_code == 403


def test_scene_ids_cannot_climb_out_of_the_output_directory(client, tmp_path):
    """/scenes globs a caller-named directory, so it must not accept a path."""
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "loot.glb").write_bytes(b"glTFloot")

    for scene_id in ("../secret", "..%2F..%2Fsecret"):
        response = client.get(f"/scenes/{scene_id}/mesh")
        assert response.status_code == 404
        assert b"loot" not in response.content


def test_unload_reports_the_model_is_gone(client):
    body = client.post("/admin/unload").json()

    assert body == {
        "model_loaded": False,
        "generators_loaded": {"hunyuan3d": False, "trellis2": False},
        "gpu": None,
    }


def test_startup_creates_the_output_directory_and_rehydrates(monkeypatch, tmp_path):
    """The startup hook is what makes results survive a restart."""
    from fastapi.testclient import TestClient

    import app
    import config
    from conftest import write_job_json

    fresh = tmp_path / "restarted"
    monkeypatch.setattr(config, "OUT_DIR", fresh)
    monkeypatch.setattr(jobs, "start_worker", lambda: None)
    fresh.mkdir()
    write_job_json(fresh, "survivor1234", jobs.DONE, created_at=1.0)
    write_job_json(fresh, "inflight1234", jobs.RUNNING, created_at=2.0)

    with TestClient(app.api) as c:
        listed = c.get("/jobs").json()["jobs"]

    assert fresh.is_dir()
    assert {j["id"]: j["status"] for j in listed} == {
        "survivor1234": jobs.DONE,
        "inflight1234": jobs.ERROR,
    }


def test_job_records_on_disk_never_contain_the_image(client, finished_job, out_dir):
    job_id = finished_job()

    raw = (out_dir / job_id / "job.json").read_text()

    assert PNG_B64 not in raw
    assert "image" not in json.loads(raw)


def test_image_providers_reports_what_is_configured(client):
    body = client.get("/images/providers").json()["providers"]

    assert {p["name"] for p in body} == {"fal", "local"}
    assert sum(p["selected"] for p in body) == 1


def test_post_images_stores_the_result_and_serves_it_back(client, monkeypatch, out_dir):
    import imagegen

    monkeypatch.setattr(
        imagegen.FalProvider, "generate", lambda self, prompt, **kw: PNG_BYTES
    )
    monkeypatch.setattr(imagegen.FalProvider, "available", lambda self: True)

    body = client.post(
        "/images", json={"prompt": "a wooden crate", "remove_background": False}
    ).json()

    assert body["provider"] == "fal"
    served = client.get(f"/images/{body['image_id']}")
    assert served.status_code == 200
    assert served.headers["content-type"] == "image/png"


def test_post_images_surfaces_a_provider_failure_as_502(client, monkeypatch):
    import imagegen

    def boom(self, prompt, **kw):
        raise imagegen.ImageGenError("fal.ai rejected the API key (401)")

    monkeypatch.setattr(imagegen.FalProvider, "generate", boom)

    response = client.post("/images", json={"prompt": "a crate"})

    assert response.status_code == 502
    assert "API key" in response.json()["detail"]


def test_post_images_rejects_an_unknown_provider(client):
    response = client.post("/images", json={"prompt": "a crate", "provider": "dalle"})

    assert response.status_code == 502
    assert "unknown image provider" in response.json()["detail"]


def test_get_image_404s_for_an_unknown_id(client):
    assert client.get("/images/nosuchimage").status_code == 404


@pytest.fixture
def fake_fal(monkeypatch):
    """A fal provider that returns a real PNG without touching the network."""
    import imagegen

    monkeypatch.setattr(imagegen.FalProvider, "available", lambda self: True)
    monkeypatch.setattr(
        imagegen.FalProvider, "generate", lambda self, prompt, **kw: PNG_BYTES
    )
    return imagegen


def test_post_image_candidates_returns_every_candidate_unchosen(client, fake_fal):
    """The choice is the point: the server never picks one."""
    body = client.post(
        "/images/candidates",
        json={"prompt": "a treasure chest", "count": 4,
              "remove_background": False},
    ).json()

    assert body["prompt"] == "a treasure chest"
    assert len(body["candidates"]) == 4
    for candidate in body["candidates"]:
        assert set(candidate) >= {"image_id", "prompt", "variant", "seed",
                                  "bytes", "path"}
        assert client.get(f"/images/{candidate['image_id']}").status_code == 200


def test_post_image_candidates_reports_what_it_spent(client, fake_fal):
    body = client.post(
        "/images/candidates",
        json={"prompt": "a crate", "count": 3, "remove_background": False},
    ).json()

    # Three images is three billed calls; nobody should learn that from a bill.
    assert body["count"] == 3
    assert "elapsed_seconds" in body


def test_post_image_candidates_defaults_to_four(client, fake_fal):
    body = client.post(
        "/images/candidates",
        json={"prompt": "a crate", "remove_background": False},
    ).json()

    assert len(body["candidates"]) == 4


def test_post_image_candidates_takes_caller_supplied_variants(client, fake_fal):
    variants = ["a plain pine chest", "a gilded rococo chest"]

    body = client.post(
        "/images/candidates",
        json={"prompt": "a chest", "count": 2, "variants": variants,
              "remove_background": False},
    ).json()

    assert [c["prompt"] for c in body["candidates"]] == variants
    assert body["mode"] == "variants"


def test_post_image_candidates_caps_the_count(client, fake_fal):
    import imagegen

    response = client.post(
        "/images/candidates",
        json={"prompt": "a crate", "count": imagegen.MAX_CANDIDATES + 1},
    )

    assert response.status_code == 422


def test_post_image_candidates_surfaces_a_provider_failure_as_502(client, monkeypatch):
    import imagegen

    def boom(self, prompt, **kw):
        raise imagegen.ImageGenError("fal.ai rejected the API key (401)")

    monkeypatch.setattr(imagegen.FalProvider, "available", lambda self: True)
    monkeypatch.setattr(imagegen.FalProvider, "generate", boom)

    response = client.post("/images/candidates", json={"prompt": "a crate"})

    assert response.status_code == 502
    assert "API key" in response.json()["detail"]


def test_a_batch_can_be_fetched_again_for_re_display(client, fake_fal):
    created = client.post(
        "/images/candidates",
        json={"prompt": "a crate", "count": 2, "remove_background": False},
    ).json()

    fetched = client.get(f"/images/batches/{created['batch_id']}")

    assert fetched.status_code == 200
    assert fetched.json() == created


def test_an_unknown_batch_404s(client):
    assert client.get("/images/batches/nosuchbatch").status_code == 404


def test_a_chosen_candidate_goes_straight_into_a_job(client, fake_fal):
    """The whole loop: generate options, choose one, generate from it."""
    batch = client.post(
        "/images/candidates",
        json={"prompt": "a crate", "count": 2, "remove_background": False},
    ).json()
    chosen = batch["candidates"][1]["image_id"]

    job = client.post("/jobs", json={"image_id": chosen}).json()
    jobs._run_one(job["id"])

    assert jobs.get(job["id"])["status"] == jobs.DONE


def test_get_primitives_lists_every_kind_with_its_parameters(client):
    body = client.get("/primitives").json()

    import primitives

    assert [k["kind"] for k in body["kinds"]] == primitives.kinds()
    assert body["max_faces"] == 20_000
    crate = next(k for k in body["kinds"] if k["kind"] == "crate")
    assert crate["material"] == "wood"
    assert {p["name"] for p in crate["params"]} >= {"width", "height", "depth"}
    assert all("default" in p for p in crate["params"])


def test_get_one_primitive_returns_just_that_kind(client):
    body = client.get("/primitives/barrel").json()

    assert body["kind"] == "barrel"
    assert body["params"]


def test_get_an_unknown_primitive_404s(client):
    assert client.get("/primitives/teapot").status_code == 404


def test_post_primitives_returns_a_finished_job_record(client):
    body = client.post("/primitives", json={"kind": "crate"}).json()

    assert body["type"] == "primitive"
    assert body["status"] == jobs.DONE
    assert body["error"] is None
    assert body["result"]["watertight"] is True
    assert body["result"]["peak_vram_gib"] == 0.0


def test_a_scripted_part_is_visible_to_the_normal_job_endpoints(client):
    job_id = client.post("/primitives", json={"kind": "barrel"}).json()["id"]

    assert client.get(f"/jobs/{job_id}").json()["id"] == job_id
    assert client.get(f"/jobs/{job_id}/mesh").status_code == 200
    assert client.get("/jobs").json()["jobs"][0]["id"] == job_id


def test_describe_reports_the_dimensions_that_were_asked_for(client):
    job_id = client.post(
        "/primitives",
        json={"kind": "crate", "params": {"width": 4.0, "height": 1.5, "depth": 2.0}},
    ).json()["id"]

    body = client.get(f"/jobs/{job_id}/describe").json()

    assert body["size"] == [4.0, 1.5, 2.0]
    assert body["center"] == [0.0, 0.0, 0.0]


def test_a_scripted_part_assembles_alongside_a_generated_one(client, finished_job):
    generated = finished_job("hull")
    scripted = client.post("/primitives", json={"kind": "wheel"}).json()["id"]

    body = client.post(
        "/assemble",
        json={
            "parts": [
                {"job_id": generated, "name": "hull"},
                {"job_id": scripted, "name": "wheel", "position": [2, 0, 0]},
            ]
        },
    ).json()

    assert body["part_count"] == 2
    assert [p["name"] for p in body["parts"]] == ["hull", "wheel"]


def test_a_scripted_part_exports_for_roblox_without_being_decimated(client):
    job_id = client.post("/primitives", json={"kind": "crate"}).json()["id"]

    body = client.post(
        "/export", json={"job_id": job_id, "target": "roblox", "height_studs": 4}
    ).json()

    assert body["total_faces"] == body["source_faces"]
    assert body["size"][1] == pytest.approx(4.0)


def test_post_primitives_mirrors_the_job_to_disk(client, out_dir):
    job_id = client.post("/primitives", json={"kind": "plank"}).json()["id"]

    saved = json.loads((out_dir / job_id / "job.json").read_text())

    assert saved["status"] == jobs.DONE
    assert saved["type"] == "primitive"


def test_post_primitives_rejects_an_unknown_kind(client):
    response = client.post("/primitives", json={"kind": "teapot"})

    assert response.status_code == 400
    assert "unknown kind" in response.json()["detail"]


def test_post_primitives_rejects_a_bad_parameter(client):
    response = client.post(
        "/primitives", json={"kind": "crate", "params": {"width": -2}}
    )

    assert response.status_code == 400
    assert "width" in response.json()["detail"]


def test_post_primitives_rejects_an_impossible_opening(client):
    response = client.post(
        "/primitives",
        json={"kind": "wall_panel", "params": {"width": 2.0, "opening_width": 2.0}},
    )

    assert response.status_code == 400


def test_post_primitives_takes_a_material_override(client):
    body = client.post(
        "/primitives", json={"kind": "crate", "material": "dark_metal"}
    ).json()

    assert body["status"] == jobs.DONE


def test_post_primitives_rejects_an_unknown_material(client):
    response = client.post("/primitives", json={"kind": "crate", "material": "cheese"})

    assert response.status_code == 400
    assert "unknown material" in response.json()["detail"]


def test_a_scripted_part_never_touches_the_queue(client):
    client.post("/primitives", json={"kind": "crate"})

    # No GPU is involved, so queueing it behind a 40-second generation would be
    # a bug rather than a policy.
    assert client.get("/health").json()["queue_depth"] == 0


def test_assembly_keeps_a_primitives_own_material(client):
    """A barrel is wood, but "barrel" reads as gun barrel — re-deriving the
    material from the node name at assembly time would turn it metal."""
    job = client.post("/primitives", json={"kind": "barrel"}).json()
    assert job["result"]["material"] == "wood"

    body = client.post(
        "/assemble", json={"parts": [{"job_id": job["id"], "name": "barrel"}]}
    ).json()

    assert body["parts"][0]["material"] == "wood"


def test_an_explicit_material_still_wins_over_the_recorded_one(client):
    job = client.post("/primitives", json={"kind": "barrel"}).json()

    body = client.post(
        "/assemble",
        json={
            "parts": [
                {"job_id": job["id"], "name": "barrel", "material": "dark_metal"}
            ]
        },
    ).json()

    assert body["parts"][0]["material"] == "dark_metal"
