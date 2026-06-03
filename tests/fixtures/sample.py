"""Module docstring for the code chunker fixture."""

import math
from dataclasses import dataclass

PI = 3.14159


def top_level_function(x: float) -> float:
    """A top-level function: should become its own function chunk."""
    return math.sqrt(x) * PI


@dataclass
class Widget:
    """A class with several methods.

    With a low max_chunk_lines this class exceeds the threshold and is split
    into a parent chunk plus one child chunk per method.
    """

    width: int
    height: int

    def area(self) -> int:
        """Return the rectangular area."""
        return self.width * self.height

    def perimeter(self) -> int:
        """Return the perimeter."""
        return 2 * (self.width + self.height)

    def scale(self, factor: int) -> "Widget":
        """Return a new Widget scaled by an integer factor."""
        return Widget(self.width * factor, self.height * factor)
