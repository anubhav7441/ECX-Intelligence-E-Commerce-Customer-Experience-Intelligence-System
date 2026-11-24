from ml.sentiment_analysis.review_nlp import analyse, normalize


def cats(t):
    return analyse(t)["complaint_categories"]


def test_normalize_strips_accents_and_case():
    assert normalize("NÃO Recebi o Produto!!!") == "nao recebi o produto!!"   # 3+ repeats are shortened to 2 by design


def test_normalize_non_string_and_empty():
    assert normalize(None) == "" and normalize(float("nan")) == "" and normalize("") == ""


def test_late_delivery_detected():
    assert "late_delivery" in cats("O produto não chegou até hoje")


def test_praise_about_deadline_is_not_a_complaint():
    assert cats("Entregue antes do prazo, amei") == ["other"]
    assert cats("Produto chegou dentro do prazo") == ["other"]


def test_damaged_missing_wrong():
    assert "damaged_product" in cats("chegou quebrado")
    assert "missing_product" in cats("Comprei dois mas só recebi um")
    assert "wrong_product" in cats("Recebi o produto errado")


def test_waiting_for_reply_is_not_late_delivery():
    assert "late_delivery" not in cats("Comprei 2 mini bike e só recebi uma, aguardo retorno.")


def test_empty_text_gives_other_and_neutral():
    r = analyse("")
    assert r["complaint_categories"] == ["other"] and r["lexicon_sentiment"] == "neutral"


def test_negation_flips_praise():
    assert analyse("não gostei, nada bom")["lexicon_sentiment"] == "negative"


def test_aspect_sentiment_positive_delivery():
    assert analyse("Entrega muito rápida")["aspects"]["delivery"] == "positive"


def test_severity_levels():
    assert analyse("Reclame aqui, golpe, nunca mais! Produto não entregue!!")["severity"] == "High"
    assert analyse("muito bom")["severity"] == "Low"
