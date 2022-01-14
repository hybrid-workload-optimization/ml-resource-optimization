import time
import queue
import threading
from collections import deque
from threading import Thread

from ..utils import MetaclassSingleton, logger


msleep = lambda x: time.sleep(x/1000.0)
usleep = lambda x: time.sleep(x/1000000.0)
nsleep = lambda x: time.sleep(x/1000000000.0)

locker = threading.Lock()

class Worker(Thread):
    _TIMEOUT = 3
    """Thread executing tasks from a given tasks queue"""
    def __init__(self, tasks, deq):
        Thread.__init__(self)
        self.tasks = tasks
        self.deq = deq
        self.done = threading.Event()
        self.log = logger.get(self.__class__.__name__)

    def run(self):
        while not self.done.is_set():
            if self.tasks.empty() is False:
                try:
                    try:
                        locker.acquire()
                        session, func, args, kargs, nowait = self.tasks.get(block=True, timeout=self._TIMEOUT)                        
                    except queue.Empty:
                        locker.release()
                        continue
                    locker.release()

                    result = func(*args, **kargs)
                    
                    if not nowait:
                        locker.acquire()
                        self.tasks.task_done()
                        self.deq.appendleft((session, result))
                        locker.release()
                
                except Exception as e:
                    self.log.warn(e)
                    
    def signal_exit(self):
        """ Signal to thread to exit """
        self.done.set()


class ThreadPool(metaclass=MetaclassSingleton):
    """Pool of threads consuming tasks from a queue"""
    def __init__(self):
        self.log = logger.get("ThreadPool")    

    def createPool(self, size):
        self.tasks = queue.Queue(128)
        self.deq = deque()
        self.session = 0
        self.workers = []
        for _ in range(size):
            self.workers.append(Worker(self.tasks, self.deq))

    def _put_task_(self, func, args, kargs, nowait):
        """Add a task to the queue"""
        self.session = self.session + 1
        self.tasks.put((self.session, func, args, kargs, nowait))
        return self.session

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue"""
        return self._put_task_(func, args, kargs, False)

    def add_task_nowait(self, func, *args, **kargs):
        """Add a task to the queue"""
        return self._put_task_(func, args, kargs, True)

    def recv(self, session):
        while True:
            #time.sleep(3)
            if len(self.deq) > 0:
                s, r = self.deq.pop()
                if s == session:
                    return r
                self.deq.appendleft((s, r))                
            nsleep(1)
    
    def wait_completion(self):
        """Wait for completion of all the tasks in the queue"""
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
                self.log.info(f"start worker thread [{i+1}]")
                worker.start()
        

    def join(self):
        for worker in self.workers:
            worker.join()

    def stop(self):
        self._close_all_threads()


