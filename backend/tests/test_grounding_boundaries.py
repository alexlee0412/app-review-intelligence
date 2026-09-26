from app.services.answer_synthesizer import _NumberToken, _is_allowed


def test_numeric_tolerance_boundary_is_strict() -> None:
    allowed = [1.0]

    assert _is_allowed(
        _NumberToken(value=1.0 + 0.5e-9, written="inside", decimal_places=None),
        allowed,
    )
    assert not _is_allowed(
        _NumberToken(value=1.0 + 1.1e-9, written="outside", decimal_places=None),
        allowed,
    )


def test_numeric_tolerance_rejects_difference_at_seventh_decimal() -> None:
    assert not _is_allowed(
        _NumberToken(value=1.0000006, written="1.0000006", decimal_places=7),
        [1.0],
    )
