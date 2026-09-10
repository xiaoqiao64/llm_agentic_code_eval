import api_client


def test_fetch_data_with_retry(monkeypatch):
    import retry as retry_mod

    api_client.fetch_data = retry_mod.retry(max_attempts=3, exceptions=(ConnectionError,))(
        api_client.fetch_data
    )
    assert api_client.fetch_data() == "ok"
