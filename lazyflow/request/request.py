###############################################################################
#   lazyflow: data flow based lazy parallel computation framework
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Lesser GNU General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# See the files LICENSE.lgpl2 and LICENSE.lgpl3 for full text of the
# GNU Lesser General Public License version 2.1 and 3 respectively.
# This information is also available on the ilastik web site at:
#		   http://ilastik.org/license/
###############################################################################
# Built-in
import sys
import heapq
import functools
import itertools
import threading
import multiprocessing
import platform
import traceback
import io
from random import randrange
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, CancelledError

import logging
logger = logging.getLogger(__name__)

import faulthandler
import signal

faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True, chain=False)


def log_exception( logger, msg=None, exc_info=None, level=logging.ERROR ):
    """
    Log the current exception to the given logger, and also log the given error message.
    If exc_info is provided, log that exception instead of the current exception provided by sys.exc_info.
    
    It is better to log exceptions this way instead of merely printing them to the console, 
    so that other logger outputs (such as log files) show the exception, too.
    """
    if sys.version_info.major == 2:
        sio = io.BytesIO()
    else:
        sio = io.StringIO()
        
    if exc_info:
        traceback.print_exception( exc_info[0], exc_info[1], exc_info[2], file=sio )
    else:
        traceback.print_exc( file=sio )

    logger.log(level, sio.getvalue() )
    if msg:
        logger.log(level, msg )

class RequestLock():
    def __init__(self):
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._lock, name)

    def __enter__(self, *args, **kwargs):
        self._lock.acquire()

    def __exit__(self, *args, **kwargs):
        self._lock.release()

class SimpleRequestCondition(threading.Condition):
    def __init__(self):
        self._condition = threading.Condition()

    def __getattr__(self, name):
        return getattr(self._condition, name)

class SimpleSignal(object):
    """
    Simple callback mechanism. Not synchronized.  No unsubscribe function.
    """
    def __init__(self, sig_lock=None):
        self.callbacks = []
        self._cleaned = False
        self.lock = sig_lock or threading.Lock()
        self.fired = False

    def subscribe(self, fn):
        with self.lock:
            if self.fired:
                fn()
            else:
                self.callbacks.append(fn)

    def __call__(self, *args, **kwargs):
        """Emit the signal."""
        assert not self._cleaned, "Can't emit a signal after it's already been cleaned!"
        with self.lock:
            if self.fired:
                return
            while self.callbacks:
                self.callbacks.pop()(*args, **kwargs)
            self.fired = True

    def clean(self):
        self._cleaned = True
        self.callbacks = []

class FakeThreadPool(object):
    num_workers = 8

class Request(object):
    NOTHING = object()
    CANCELLED = object()

    REQUEST_THREAD_PREFIX="REQUEST_THREAD_"
    executor = ThreadPoolExecutor(max_workers=99999999, thread_name_prefix=REQUEST_THREAD_PREFIX)

    global_thread_pool = FakeThreadPool()

    active_count = 0

    class CancellationException(Exception):
        pass

    class InvalidRequestException(Exception):
        pass

    class CircularWaitException(Exception):
        pass
    
    class TimeoutException(Exception):
        pass
    
    class InternalError(Exception):
        pass

    def __init__(self, fn, root_priority=[0], on_fail=(), on_cancel=(), on_finish=(), on_complete=()):
        self.lock = threading.Lock()
        self._sig_failed = SimpleSignal(self.lock)
        self._sig_cancelled = SimpleSignal(self.lock)
        self._sig_finished = SimpleSignal(self.lock)
        self._sig_execution_complete = SimpleSignal(self.lock)

        self._result = self.NOTHING

        # Workload
        self.fn = fn
        self._future = None

    def __call__(self):
        try:
            self._result = self.fn()
            #import pydevd; pydevd.settrace()
            self._sig_finished(self._result)
            return self._result
        except CancelledError as ce:
            self._sig_cancelled()
            raise Request.CancellationException from ce
        except Exception as e:
            self._sig_failed(e, sys.exc_info())
            raise e
        finally:
            self.active_count -= 1
            self._sig_execution_complete()

    def clean(self, _fullClean=True):
        pass

    def submit(self):
        with self.lock:
            if self._future is None:
                self._future = self.executor.submit(self)
                self.active_count += 1

    @property
    def result(self):
        return self.wait()

    def wait(self, timeout=None):
        self.submit()
        return self._future.result(timeout)

    def block(self, timeout=None):
        self.wait()

    def notify_finished(self, fn):
        self._sig_finished.subscribe(fn)

    def notify_cancelled(self, fn):
        self._sig_cancelled.subscribe(fn)

    def notify_failed(self, fn):
        self._sig_failed.subscribe(fn)

    def cancel(self):
        self._future.cancel()

    class _PartialWithAppendedArgs(object):
        """
        Like functools.partial, but any kwargs provided are given last when calling the target.
        """
        def __init__(self, fn, *args, **kwargs):
            self.func = fn
            self.args = args
            self.kwargs = kwargs
        
        def __call__(self, *args):
            totalargs = args + self.args
            return self.func( *totalargs, **self.kwargs)
    
    def writeInto(self, destination):
        self.fn = Request._PartialWithAppendedArgs( self.fn, destination=destination )
        return self



    def getResult(self):
        return self.result

#def doNothing():
#    import time
#    time.sleep(9000)
#
#threadCount = 0
#while True:
#    Request(doNothing).submit()
#    threadCount += 1
#    print(f"spawned {threadCount} threads")

class Request_(Request):
    def submit(self):
        pass

    def wait():
        if self._result is self.NOTHING:
            self()
        return self._result


class RequestPool(object):
    class RequestPoolError(Exception):
        pass

    def __init__(self, max_active=None):
        pass

    def clean(self):
        pass

    def __len__(self):
        #return len(self._unsubmitted_requests) + len(self._active_requests) + len(self._finishing_requests)
        return len(self._unsubmitted_requests) + len(self._active_requests) + len(self._finishing_requests)

    def add(self, req):
        pass

    def wait(self):
        pass

    def cancel(self):
        pass

    def request(self, func):
        return Request(func)
