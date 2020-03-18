import unittest
from python.utils import (
    convert_camel_to_snake_case,
    convert_snake_to_camel_case
)

class ConvertCamelToSnakeCaseTest(unittest.TestCase):
    def test_convert_empty_string(self):
        out = convert_camel_to_snake_case("")
        self.assertEqual("", out)

    def test_convert_single_alphanumerical_word(self):
        out = convert_camel_to_snake_case("word")
        self.assertEqual("word", out)

    def test_convert_single_nonalphanumerical_word(self):
        out = convert_camel_to_snake_case("word_word")
        self.assertEqual("word_word", out)

    def test_convert_two_words_first_lowercase(self):
        out = convert_camel_to_snake_case("testTest")
        self.assertEqual("test_test", out)

    def test_convert_two_words_first_uppercase(self):
        out = convert_camel_to_snake_case("FooBar")
        self.assertEqual("foo_bar", out)

    def test_convert_two_words_non_alphanumerical(self):
        out = convert_camel_to_snake_case("Foo_Bar")
        self.assertEqual("foo__bar", out)

class ConvertSnakeCaseToCamelCaseTest(unittest.TestCase):
    def test_convert_empty_string(self):
        out = convert_snake_to_camel_case("")
        self.assertEqual("", out)

    def test_convert_single_alphanumerical_word(self):
        out = convert_snake_to_camel_case("word")
        self.assertEqual("word", out)

    def test_convert_single_nonalphanumerical_word(self):
        out = convert_snake_to_camel_case("word.word")
        self.assertEqual("word.word", out)

    def test_convert_two_words(self):
        out = convert_snake_to_camel_case("test_test")
        self.assertEqual("testTest", out)

    def test_convert_two_words_mixed_case(self):
        out = convert_snake_to_camel_case("Foo_bar")
        self.assertEqual("fooBar", out)

    def test_convert_two_words_non_alphanumerical(self):
        out = convert_snake_to_camel_case("Foo.Bar_xyz")
        self.assertEqual("foo.BarXyz", out)


if __name__ == "__main__":
    unittest.main()



