# Exercise 3: Temperature Converter
from typing import Union

# Defining a Temperature type for cleaner type hints
Temperature = Union[int, float]


def celsius_to_fahrenheit(celsius: Temperature) -> float:
    """
    Convert temperature from Celsius to Fahrenheit.

    Formula: F = C × 9/5 + 32

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Fahrenheit (rounded to 2 decimal places)
    """
    return (celsius*(9/5))+32

def fahrenheit_to_celsius(fahrenheit: Temperature) -> float:
    """
    Convert temperature from Fahrenheit to Celsius.

    Formula: C = (F - 32) × 5/9

    Args:
        fahrenheit: Temperature in Fahrenheit

    Returns:
        Temperature in Celsius (rounded to 2 decimal places)
    """
    return (fahrenheit-32)*(5/9)


def celsius_to_kelvin(celsius: Temperature) -> float:
    """
    Convert temperature from Celsius to Kelvin.

    Formula: K = C + 273.15

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Kelvin (rounded to 2 decimal places)
    """
    return celsius + 273.15


def kelvin_to_celsius(kelvin: Temperature) -> float:
    """
    Convert temperature from Kelvin to Celsius.

    Formula: C = K - 273.15

    Args:
        kelvin: Temperature in Kelvin

    Returns:
        Temperature in Celsius (rounded to 2 decimal places)

    Raises:
        ValueError: If kelvin is less than 0 (below absolute zero)
    """
    return kelvin - 273.15
