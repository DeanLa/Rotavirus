from datetime import datetime
import logging
from datetime import timedelta
import re

# from numeric_utils import trim_to_first_significant_digits
logger = logging.getLogger(__name__)


class _elapsed_time(object):
    def __init__(self, message="Time Elapsed"):
        self.message = message

    def __enter__(self):
        self.stopper = Stopwatch()

    def __exit__(self, ex_type, value, traceback):
        elapsed_time = "{0}: {1}".format(self.message, self.stopper.end_now())
        self.output_elapsed_time(elapsed_time)
    @staticmethod
    def output_elapsed_time(elapsed_time):
        raise NotImplementedError()


class LogElapsedTime(_elapsed_time):
    @staticmethod
    def output_elapsed_time(elapsed_time):
        logger.info(elapsed_time)


class PrintElapsedTime(_elapsed_time):
    @staticmethod
    def output_elapsed_time(elapsed_time):
        print(elapsed_time)


class Stopwatch(object):
    def __init__(self):
        self.start_time = datetime.now()

    def end_now(self):
        return (datetime.now() - self.start_time).total_seconds()


def longer_time_string(run_time_in_seconds):
    time_delta = timedelta(seconds=run_time_in_seconds)
    timedelta_string = str(time_delta)

    m = re.search('(\d* (days|day), )?(\d*):(\d*):(\d*)', timedelta_string)
    days_string = m.group(1)
    hours = int(m.group(3))
    minutes = int(m.group(4))
    seconds = int(m.group(5))

    if days_string:
        days_string = days_string[:-2]
        return "{}, {} hours, {} minutes, {} seconds".format(days_string, hours, minutes, seconds)
    elif hours:
        return "{} hours, {} minutes, {} seconds".format(hours, minutes, seconds)
    elif minutes:
        return "{} minutes, {} seconds".format(minutes, seconds)
    else:
        return "{} seconds".format(seconds)

# def get_time_string(time):
#     if time > 1:
#         time_string = "{:.1f} seconds".format(time)
#         if time > 60:
#             time_string = longer_time_string(time)
#     else:
#         time *= 1000
#         if time > 1:  # milliseconds scale
#             time_string = "{} seconds ({:.0f} milliseconds)".format(
#                 trim_to_first_significant_digits(time/10**3, significant_digits=1), time)
#         else:
#             time *= 1000
#             time_string = "{} seconds ({:.0f} microseconds)".format(
#                 trim_to_first_significant_digits(time/10**6, significant_digits=1), time)
#
#     return time_string
