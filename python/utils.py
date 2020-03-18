import re

def convert_camel_to_snake_case(string: str):
    """
    Converts a string written in camelCase to a string written in a snake_case.

    :param string: a camelCase string to convert
    :return: a string converted to snake_case
    """
    words = re.findall("[a-zA-Z][^A-Z]*", string)
    words = (w.lower() for w in words)
    return "_".join(words)


def convert_snake_to_camel_case(string: str):
    """
    Converts a string written in snake_case to a string written in a camelCase.

    The output string will start with lower case letter.

    :param string: a snake_case string to convert
    :return: a string converted to camelCase
    """
    words = string.split("_")
    capitalized_words = (w.capitalize() for w in words[1:])
    first_word = words[0]
    first_word = first_word[:1].lower() + first_word[1:]
    words = [first_word]
    words.extend(capitalized_words)
    return "".join(words)
