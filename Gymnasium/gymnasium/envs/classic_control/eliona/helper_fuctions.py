import holidays
from datetime import datetime


def datetime_to_time_attributes(dt = None,country="CH"):
    """
    Convert a datetime object into a dictionary of time attributes, including holiday information.

    :param dt: A datetime object.
    :param country: The country code for holiday determination (default: US).
    :return: A dictionary containing time attributes.
    """
    if dt is None:
        dt = datetime.now()
    if not isinstance(dt, datetime):
        raise ValueError("Input must be a datetime object")

    # Load holidays for the specified country
    holiday_calendar = holidays.country_holidays(country)

    time_attributes = {
        "time_of_day": dt.hour + dt.minute / 60 + dt.second / 3600,  # Fractional hours
        "day_of_week": dt.weekday(),
        "day_of_month": dt.day,
        "month_of_year": dt.month,
        "week_of_year": dt.isocalendar()[1],
        "year": dt.year,
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "is_holiday": 1 if dt in holiday_calendar else 0,
        "minute_of_hour": dt.minute,
    }
    return time_attributes
