import csv
import io
import threading
import time

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPLY_DB", str(tmp_path / "supply.db"))
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def publish(client, batch="b1", tickets="", parts="", row_limit="10"):
    return client.post(
        f"/api/batches/{batch}/publish",
        files={
            "tickets": ("tickets.csv", tickets.encode(), "text/csv"),
            "parts": ("parts.csv", parts.encode(), "text/csv"),
        },
        data={"row_limit": row_limit},
    )


def rows(text):
    return list(csv.reader(io.StringIO(text)))


TICKETS = (
    "ticket_no,station\n"
    "T3,S3\n"
    "T1,S1\n"
    "T2,S2\n"
    "T2,S2\n"
)
PARTS = (
    "ticket_no,part_code\n"
    "T1,P1\n"
    "T2,P2\n"
    "T2,P2\n"
)


def test_publish_returns_manifest_with_package_boundaries(client):
    response = publish(client, tickets=TICKETS, parts=PARTS, row_limit="10")
    assert response.status_code == 201, response.text
    manifest = response.json()
    assert manifest["version"] == "b1.v0001"
    assert manifest["row_limit"] == 10
    # T1 1x1, T2 2x2, T3 1x0 -> 1 + 4 + 1
    assert manifest["total_rows"] == 6
    assert manifest["package_count"] == 1
    package = manifest["packages"][0]
    assert package == {
        "package_no": 1,
        "first_ticket_no": "T1",
        "last_ticket_no": "T3",
        "row_count": 6,
        "download_url": "/api/manifests/b1.v0001/packages/1.csv",
    }


def test_download_package_streams_joined_rows_in_ticket_order(client):
    manifest = publish(client, tickets=TICKETS, parts=PARTS, row_limit="10").json()
    response = client.get(manifest["packages"][0]["download_url"])
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert rows(response.text) == [
        ["ticket_no", "station", "part_code"],
        ["T1", "S1", "P1"],
        ["T2", "S2", "P2"],
        ["T2", "S2", "P2"],
        ["T2", "S2", "P2"],
        ["T2", "S2", "P2"],
        ["T3", "S3", ""],
    ]


def test_pack_boundaries_and_groups_not_split(client):
    tickets = "ticket_no,station\n" + "".join(
        f"{ticket},S\n" for ticket in ["T1", "T2", "T3", "T4"]
    )
    parts = "ticket_no,part_code\n" + "T1,A\nT1,B\nT2,C\nT2,D\nT3,E\n"
    # groups: T1=2, T2=2, T3=1, T4=1(no parts) -> limit 3:
    # pkg1 T1 (2) + T2 would be 4 -> pkg1=T1; pkg2=T2(2)+T3 would be 3 -> T2,T3; pkg3=T4
    manifest = publish(client, tickets=tickets, parts=parts, row_limit="3").json()
    assert [
        (p["package_no"], p["first_ticket_no"], p["last_ticket_no"], p["row_count"])
        for p in manifest["packages"]
    ] == [(1, "T1", "T1", 2), (2, "T2", "T3", 3), (3, "T4", "T4", 1)]

    for package in manifest["packages"]:
        body = rows(client.get(package["download_url"]).text)
        assert len(body) - 1 == package["row_count"]


def test_oversized_ticket_reports_numbers_and_publishes_nothing(client):
    first = publish(client, tickets=TICKETS, parts=PARTS, row_limit="10")
    assert first.status_code == 201
    original_version = first.json()["version"]

    response = publish(client, tickets=TICKETS, parts=PARTS, row_limit="3")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "ticket_row_limit_exceeded"
    assert detail["ticket_no"] == "T2"
    assert detail["row_count"] == 4
    assert detail["row_limit"] == 3

    # Failed resubmission leaves the original manifest as the latest one.
    latest = client.get("/api/batches/b1/manifests/latest").json()
    assert latest["version"] == original_version
    assert [
        m["version"] for m in client.get("/api/batches/b1/manifests").json()["manifests"]
    ] == [original_version]
    assert client.get(f"/api/manifests/{original_version}/packages/1.csv").status_code == 200


def test_manifest_listing_does_not_expand_details(client):
    publish(client, tickets=TICKETS, parts=PARTS, row_limit="10")
    listing = client.get("/api/batches/b1/manifests")
    assert listing.status_code == 200
    body = listing.json()
    assert body["batch_id"] == "b1"
    for manifest in body["manifests"]:
        assert "detail_rows" not in manifest
        for package in manifest["packages"]:
            assert set(package) == {
                "package_no",
                "first_ticket_no",
                "last_ticket_no",
                "row_count",
                "download_url",
            }

    csv_response = client.get("/api/manifests/b1.v0001/manifest.csv")
    assert csv_response.status_code == 200
    table = rows(csv_response.text)
    assert table[0] == [
        "version",
        "package_no",
        "first_ticket_no",
        "last_ticket_no",
        "row_count",
    ]
    assert table[1] == ["b1.v0001", "1", "T1", "T3", "6"]


def test_resubmission_same_batch_creates_new_version(client):
    v1 = publish(client, tickets=TICKETS, parts=PARTS, row_limit="10").json()
    v2 = publish(
        client,
        tickets=TICKETS,
        parts=PARTS + "T1,P1B\n",
        row_limit="10",
    ).json()
    assert v1["version"] == "b1.v0001"
    assert v2["version"] == "b1.v0002"
    # T1 now has two part lines for one ticket row: 2 + 4 + 1.
    assert v2["total_rows"] == 7

    versions = [
        m["version"] for m in client.get("/api/batches/b1/manifests").json()["manifests"]
    ]
    assert versions == ["b1.v0002", "b1.v0001"]


def test_old_manifest_still_downloads_after_new_publish(client):
    v1 = publish(client, tickets=TICKETS, parts=PARTS, row_limit="10").json()
    old_rows = rows(client.get(v1["packages"][0]["download_url"]).text)
    assert len(old_rows) == 7

    v2 = publish(client, tickets=TICKETS, parts=PARTS + "T1,P1B\n", row_limit="10").json()
    assert v2["version"] != v1["version"]
    new_rows = rows(client.get(v2["packages"][0]["download_url"]).text)
    assert len(new_rows) == 8

    # The old version is immutable and still served from its own snapshot.
    assert rows(client.get(v1["packages"][0]["download_url"]).text) == old_rows
    assert client.get("/api/manifests/b1.v0001").json()["total_rows"] == 6
    assert client.get("/api/batches/b1/manifests/latest").json()["version"] == v2["version"]


def test_download_reads_only_the_requested_package(client):
    tickets = "ticket_no,station\nT1,S1\nT2,S2\nT3,S3\nT4,S4\n"
    parts = "ticket_no,part_code\nT1,P1\nT2,P2\nT3,P3\nT4,P4\n"
    manifest = publish(client, tickets=tickets, parts=parts, row_limit="2").json()
    assert [p["row_count"] for p in manifest["packages"]] == [2, 2]

    first = rows(client.get(manifest["packages"][0]["download_url"]).text)
    second = rows(client.get(manifest["packages"][1]["download_url"]).text)
    assert first[1:] == [["T1", "S1", "P1"], ["T2", "S2", "P2"]]
    assert second[1:] == [["T3", "S3", "P3"], ["T4", "S4", "P4"]]


def test_unknown_version_or_package_is_404(client):
    publish(client, tickets=TICKETS, parts=PARTS, row_limit="10")
    assert client.get("/api/manifests/nope").status_code == 404
    assert client.get("/api/manifests/b1.v0001/packages/99.csv").status_code == 404
    assert client.get("/api/manifests/nope/packages/1.csv").status_code == 404


def test_bad_csv_is_rejected(client):
    response = publish(client, tickets="ticket_no\nT1\n", parts="", row_limit="10")
    assert response.status_code == 422
    response = publish(client, tickets="ticket_no,station\nT1,S1\n", parts="", row_limit="0")
    assert response.status_code == 422


def test_concurrent_download_while_publishing_new_version(client):
    tickets = "ticket_no,station\n" + "".join(
        f"T{i:03d},S\n" for i in range(50)
    )
    parts = "ticket_no,part_code\n" + "".join(f"T{i:03d},P\n" for i in range(50))
    v1 = publish(client, tickets=tickets, parts=parts, row_limit="10").json()
    expected = {
        p["package_no"]: rows(client.get(p["download_url"]).text)
        for p in v1["packages"]
    }

    errors = []

    def download_all():
        try:
            for package_no, expected_rows in expected.items():
                body = rows(
                    client.get(
                        f"/api/manifests/{v1['version']}/packages/{package_no}.csv"
                    ).text
                )
                assert body == expected_rows
        except Exception as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    worker = threading.Thread(target=download_all)
    worker.start()
    v2 = publish(client, tickets=tickets, parts=parts + "T000,P2\n", row_limit="10")
    worker.join()
    assert not errors
    assert v2.status_code == 201
    assert v2.json()["version"] == "b1.v0002"


def test_chinese_column_headers_are_accepted(client):
    response = publish(
        client,
        tickets="工单号,送修点\nW1,SP1\nW2,SP2\n",
        parts="工单号,零件码\nW1,PN1\n",
        row_limit="10",
    )
    assert response.status_code == 201, response.text
    manifest = response.json()
    body = rows(client.get(manifest["packages"][0]["download_url"]).text)
    assert body[1:] == [
        ["W1", "SP1", "PN1"],
        ["W2", "SP2", ""],
    ]


def test_parts_file_without_data_rows_keeps_every_ticket(client):
    response = publish(
        client,
        tickets="ticket_no,station\nT1,S1\nT1,S1\nT2,S2\n",
        parts="ticket_no,part_code\n",
        row_limit="3",
    )
    assert response.status_code == 201, response.text
    manifest = response.json()
    # T1 (2 duplicate rows) + T2 (1 row) = 3 rows, exactly the limit.
    assert [p["row_count"] for p in manifest["packages"]] == [3]
    body = rows(client.get(manifest["packages"][0]["download_url"]).text)
    assert body[1:] == [
        ["T1", "S1", ""],
        ["T1", "S1", ""],
        ["T2", "S2", ""],
    ]


def test_quoting_and_commas_survive_round_trip(client):
    response = publish(
        client,
        tickets='ticket_no,station\nW1,"SP,1"\n',
        parts='ticket_no,part_code\nW1,"PN ""A"""\n',
        row_limit="10",
    )
    assert response.status_code == 201, response.text
    manifest = response.json()
    body = rows(client.get(manifest["packages"][0]["download_url"]).text)
    assert body[1] == ["W1", "SP,1", 'PN "A"']


def test_header_only_ticket_csv_is_rejected(client):
    response = publish(
        client, tickets="ticket_no,station\n", parts="ticket_no,part_code\n"
    )
    assert response.status_code == 422
    assert "no data rows" in response.json()["detail"]


def test_whitespace_padded_rows_are_trimmed(client):
    response = publish(
        client,
        tickets="  ticket_no , station \n T1 , S1 \n",
        parts="ticket_no,part_code\nT1, P1 \n",
        row_limit="10",
    )
    assert response.status_code == 201, response.text
    body = rows(
        client.get(response.json()["packages"][0]["download_url"]).text
    )
    assert body[1] == ["T1", "S1", "P1"]


def test_concurrent_publishes_are_serialized_per_batch(client):
    errors = []
    published = {}

    def submit(index):
        try:
            response = publish(
                client,
                tickets="ticket_no,station\nT1,S1\n",
                parts=f"ticket_no,part_code\nT1,P{index}\n",
                row_limit="10",
            )
            assert response.status_code == 201, response.text
            published[index] = response.json()["version"]
        except Exception as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    workers = [threading.Thread(target=submit, args=(i,)) for i in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert not errors
    versions = [
        m["version"] for m in client.get("/api/batches/b1/manifests").json()["manifests"]
    ]
    assert sorted(versions) == [f"b1.v{i:04d}" for i in range(1, 7)]
    # Every accepted publish got a distinct version, and each version serves
    # exactly the snapshot that request submitted.
    assert len(set(published.values())) == 6
    for index, version in published.items():
        body = rows(
            client.get(f"/api/manifests/{version}/packages/1.csv").text
        )
        assert body[1] == ["T1", "S1", f"P{index}"]


def test_download_does_not_include_other_batches_same_ticket_numbers(client):
    publish(client, batch="b1", tickets="ticket_no,station\nT1,S1\n",
            parts="ticket_no,part_code\nT1,P1\n", row_limit="10")
    publish(client, batch="b2", tickets="ticket_no,station\nT1,S1\n",
            parts="ticket_no,part_code\nT1,P2\n", row_limit="10")

    v1 = client.get("/api/batches/b1/manifests/latest").json()
    v2 = client.get("/api/batches/b2/manifests/latest").json()
    assert rows(client.get(v1["packages"][0]["download_url"]).text)[1] == ["T1", "S1", "P1"]
    assert rows(client.get(v2["packages"][0]["download_url"]).text)[1] == ["T1", "S1", "P2"]


def test_slow_download_clients_do_not_block_publish(client):
    # A package big enough to keep every stalled stream suspended on
    # backpressure well beyond the publish below.
    tickets = "ticket_no,station\n" + "".join(
        f"T{i:05d},S\n" for i in range(3000)
    )
    parts = "ticket_no,part_code\n" + "".join(
        f"T{i:05d},P\n" for i in range(3000)
    )
    manifest = publish(client, tickets=tickets, parts=parts, row_limit="3000").json()
    url = manifest["packages"][0]["download_url"]

    stall = threading.Event()
    release = threading.Event()
    errors = []

    def stalled_download():
        try:
            with client.stream("GET", url) as response:
                assert response.status_code == 200
                first = True
                for _ in response.iter_bytes():
                    if first:
                        # Read the first chunk, then stop draining entirely:
                        # the server side stays suspended on backpressure.
                        stall.set()
                        first = False
                        release.wait(timeout=30)
                        if not release.is_set():
                            raise RuntimeError("stall timed out")
        except Exception as exc:  # pragma: no cover - re-raised below
            errors.append(exc)

    # More concurrent stalls than the anyio default thread limiter (40).
    workers = [threading.Thread(target=stalled_download) for _ in range(45)]
    for worker in workers:
        worker.start()
    assert stall.wait(timeout=15), "slow clients never started streaming"
    time.sleep(0.5)  # let every stream settle onto backpressure

    # With sync streaming each stalled download keeps a worker pinned inside
    # the SQLite cursor; the resulting convoy makes a publish take seconds
    # (and on a single-loop uvicorn server it can be starved indefinitely).
    # The async streamer only borrows a worker for bounded, lock-serialized
    # batches, so the publish must finish promptly while all 45 downloads are
    # still suspended.
    def publish_while_stalled():
        started = time.monotonic()
        response = publish(client, tickets=tickets, parts=parts + "T00000,P2\n",
                           row_limit="3000")
        errors.append((response.status_code, time.monotonic() - started))

    publisher = threading.Thread(target=publish_while_stalled)
    publisher.start()
    publisher.join(timeout=15)
    assert not publisher.is_alive(), "publish starved by slow downloads"

    release.set()
    for worker in workers:
        worker.join(timeout=15)
        assert not worker.is_alive()

    assert len(errors) == 1, errors
    status_code, elapsed = errors[0]
    assert status_code == 201, errors
    # Old synchronous streaming measured 6+ seconds here; the fixed design
    # stays well under that because waiting for the DB happens on the event
    # loop instead of inside pinned worker threads.
    assert elapsed < 3.0, f"publish took {elapsed:.2f}s under stalled downloads"

    listing = client.get("/api/batches/b1/manifests").json()["manifests"]
    assert [m["version"] for m in listing] == ["b1.v0002", "b1.v0001"]


def test_started_download_completes_while_publish_transaction_is_open(client):
    from app import db as db_module

    tickets = "ticket_no,station\n" + "".join(
        f"T{i:05d},S\n" for i in range(1500)
    )
    parts = "ticket_no,part_code\n" + "".join(
        f"T{i:05d},P\n" for i in range(1500)
    )
    manifest = publish(client, tickets=tickets, parts=parts, row_limit="1500").json()
    url = manifest["packages"][0]["download_url"]

    # Hold an open, uncommitted writer transaction on the same database the
    # whole time the download is being drained.
    writer = db_module.connect()
    writer.execute("BEGIN IMMEDIATE")
    try:
        writer.execute(
            "INSERT INTO manifests (version, batch_id, seq, row_limit,"
            " total_rows, ticket_count, package_count, created_at)"
            " VALUES ('b1.v9999', 'b1', 9999, 1, 1, 1, 1, 'x')"
        )
        assert writer.in_transaction

        body = client.get(url).text
    finally:
        writer.execute("ROLLBACK")
        writer.close()

    table = rows(body)
    assert len(table) == 1501
    assert table[0] == ["ticket_no", "station", "part_code"]
    assert table[1] == ["T00000", "S", "P"]
    assert table[-1] == ["T01499", "S", "P"]
    # Only the pre-existing published version is visible; the uncommitted
    # write never leaked into the read.
    assert client.get("/api/manifests/b1.v9999").status_code == 404
    assert client.get("/api/batches/b1/manifests/latest").json()["version"] == "b1.v0001"


def test_failed_publish_keeps_latest_successful_manifest(client):
    first = publish(client, tickets=TICKETS, parts=PARTS, row_limit="10")
    assert first.status_code == 201
    good_version = first.json()["version"]

    # Oversized ticket: validation rejects the batch before any write.
    response = publish(client, tickets=TICKETS, parts=PARTS, row_limit="3")
    assert response.status_code == 422

    latest = client.get("/api/batches/b1/manifests/latest")
    assert latest.status_code == 200
    assert latest.json()["version"] == good_version
    body = rows(client.get(
        f"/api/manifests/{good_version}/packages/1.csv"
    ).text)
    assert body[0] == ["ticket_no", "station", "part_code"]
    assert len(body) - 1 == latest.json()["total_rows"]
