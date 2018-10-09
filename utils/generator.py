import threading

class threadsafe_iter:
    """Takes a generator and makes it threadsafe.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator for  make a generator threadsafe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g