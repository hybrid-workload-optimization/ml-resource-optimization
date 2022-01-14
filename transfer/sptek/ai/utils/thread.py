import queue
import threading
import time

from collections import deque
from itertools import count
from threading import Thread

from ..utils import logger
from ..utils import MetaclassSingleton


msleep = lambda x: time.sleep(x/1000.0)
usleep = lambda x: time.sleep(x/1000000.0)
nsleep = lambda x: time.sleep(x/1000000000.0)

locker = threading.Lock()

class Worker(Thread):
    _TIMEOUT = 3
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks, commit):
        Thread.__init__(self)
        self.tasks = tasks
        self.commit = commit
        self.done = threading.Event()
        self.flag = False
        self.log = logger.get(self.__class__.__name__)

    def run(self):
        while not self.done.is_set():
            if self.tasks.empty() is False:
                try:
                    
                    try:
                        session, func, args, kargs, nowait = self.tasks.get(block=True, timeout=self._TIMEOUT)
                    except queue.Empty:
                        continue

                    self.flag = True

                    kargs['session'] = session
                    result = func(*args, **kargs)
                    
                    if not nowait:
                        self.tasks.task_done()
                        self.commit.append((session, result))

                    self.flag = False
                
                except Exception as e:
                    self.log.warn(e)

    def is_active(self):
        return self.flag
                    
    def signal_exit(self):
        """ Signal to thread to exit """
        self.done.set()


class ThreadPool(metaclass=MetaclassSingleton):
    """Pool of threads consuming tasks from a queue"""
    def __init__(self):
        self.log = logger.get("ThreadPool")    

    def createPool(self, size):
        self.tasks = queue.Queue(128)
        self.commit = []
        self.session = 0
        self.workers = []
        for _ in range(size):
            self.workers.append(Worker(self.tasks, self.commit))

    def _put_task_(self, func, args, kargs, nowait):
        """Add a task to the queue"""
        locker.acquire()        
        self.session = self.session + 1
        _session_ = self.session
        locker.release()
        self.tasks.put((_session_, func, args, kargs, nowait))
        return _session_

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        return self._put_task_(func, args, kargs, False)

    def add_task_nowait(self, func, *args, **kargs):
        """Add a task to the queue"""
        return self._put_task_(func, args, kargs, True)

    def recv(self, session):
        while True:
            for item in self.commit:
                if item[0] == session:
                    self.commit.remove(item)
                    return item[1]
            nsleep(1)

    def _wait_completion_session_(self, session):
        return self.recv(session)
    
    def wait_completion(self, session=None):
        """Wait for completion of all the tasks in the queue"""
        if session is not None:
            return self._wait_completion_session_(session)
        self.tasks.join()

    def _close_all_threads(self):
        """ Signal all threads to exit and lose the references to them """
        for worker in self.workers:
            worker.signal_exit()
        self.workers = []
        self.log.info(f"stop the all threads.")

    def __del__(self):
        self._close_all_threads()

    def start(self):
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():                
                worker.start()
        self.log.info(f"start worker thread [1 ~ {i+1}]")

    def join(self):
        for worker in self.workers:
            worker.join()

    def stop(self):
        self._close_all_threads()

    def active_task(self):
        count = 0
        for worker in self.workers:
            if worker.is_active():
                count = count + 1
        return count


