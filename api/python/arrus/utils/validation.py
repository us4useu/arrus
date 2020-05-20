

def assert_not_none(value, value_name: str):
    if value is None:
        raise ValueError("Value '%s' is required.")