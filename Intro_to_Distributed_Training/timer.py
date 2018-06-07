# Code from https://github.com/miguelgfierro/codebase/blob/master/python/log_base/timer.py
from timeit import default_timer
from datetime import timedelta


class Timer(object):
    """Timer class.
    Examples:
        >>> import numpy as np
        >>> big_num = 10000000
        >>> t = Timer()
        >>> t.start()
        >>> r = 0
        >>> a = [r+i for i in range(big_num)]
        >>> t.stop()
        >>> np.round(t.interval)
        2.0
        >>> r = 0
        >>> with Timer() as t:
        ...   a = [r+i for i in range(big_num)]
        >>> np.round(t.interval)
        2.0
        >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        'Time elapsed 0:00:...'

    """
    def __init__(self):
        self._timer = default_timer
        self.interval = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return str(timedelta(seconds=self.interval))

    def start(self):
        """Start the timer."""
        self.init = self._timer()

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        self.interval = self.end - self.init

