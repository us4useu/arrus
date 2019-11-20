
def assert_true(value: bool, desc: str = ""):
    if not value:
        msg = "Assertion false"
        if desc:
            msg += ": " % desc
        raise ValueError(msg)