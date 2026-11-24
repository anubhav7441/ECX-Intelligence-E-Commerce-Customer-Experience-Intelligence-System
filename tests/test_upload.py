import pytest
from app.services.upload_service import UploadError, validate_csv


def test_valid_orders_like_csv():
    csv = b"seller_id,seller_zip_code_prefix,seller_city,seller_state\na,1,x,SP\nb,2,y,RJ\n"
    r = validate_csv("s.csv", csv)
    assert r["rows"] == 2 and r["detected_table"] == "sellers" and r["quality"] == "OK"


def test_wrong_extension():
    with pytest.raises(UploadError):
        validate_csv("data.xlsx", b"a,b\n1,2\n")


def test_empty_file():
    with pytest.raises(UploadError):
        validate_csv("a.csv", b"")


def test_header_only_file():
    with pytest.raises(UploadError):
        validate_csv("a.csv", b"a,b,c\n")


def test_binary_garbage_rejected_or_flagged():
    try:
        r = validate_csv("a.csv", bytes(range(256)) * 20)
        assert r["detected_table"] is None
    except UploadError:
        pass


def test_duplicates_and_missing_reported():
    csv = b"seller_id,seller_zip_code_prefix,seller_city,seller_state\na,1,,SP\na,1,,SP\n"
    r = validate_csv("s.csv", csv)
    assert r["duplicate_rows"] == 1 and r["duplicate_keys"] == 1 and r["quality"] == "ISSUES FOUND"
    assert any(m["column"] == "seller_city" for m in r["missing_values"])


def test_invalid_review_score_and_dates_flagged():
    csv = (b"review_id,order_id,review_score,review_comment_title,review_comment_message,review_creation_date,review_answer_timestamp\n"
           b"r1,o1,9,,,not-a-date,2018-01-01 00:00:00\n")
    r = validate_csv("r.csv", csv)
    assert any("review_score" in i for i in r["issues"]) and any("not valid dates" in i for i in r["issues"])


def test_unknown_table_still_profiled():
    r = validate_csv("x.csv", b"foo,bar\n1,2\n")
    assert r["detected_table"] is None and r["rows"] == 1


def test_oversize_rejected():
    with pytest.raises(UploadError):
        validate_csv("big.csv", b"a\n" + b"1\n" * (11 * 1024 * 1024))
