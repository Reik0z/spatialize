import logging
from rich.logging import RichHandler
from rich.progress import Progress
from tqdm.auto import tqdm


class progress:
    init = "init"
    step = "step"
    token = "token"
    done = "done"

    prog = "progress"

    @classmethod
    def init(cls, total, increment_step=1):
        return {cls.prog: {cls.init: total, cls.step: increment_step}}

    @classmethod
    def stop(cls):
        return {cls.prog: cls.done}

    @classmethod
    def inform(cls, value=None):
        return {cls.prog: {cls.token: value}}


class AsyncProgressHandler:
    def __init__(self):
        self.total = None
        self.step = None
        self._reset()

    def _done(self):
        raise NotImplemented

    def _update(self):
        raise NotImplemented

    def __call__(self, msg):
        if not self._pass_protocol(msg):
            return
        if isinstance(msg[progress.prog], dict):
            if progress.init in msg[progress.prog]:
                self._init(msg[progress.prog][progress.init], msg[progress.prog][progress.step])
            if progress.token in msg[progress.prog]:
                # no need to process the incoming value
                self._increment()
                self._update()
        if msg[progress.prog] == progress.done:
            self._done()
            self._reset()

    def _init(self, total, step):
        self.total = total
        self.step = step

    def _reset(self):
        self.count = 0
        self.p = 0
        self.p_prev = -1

    def _increment(self):
        self.count += self.step

    @staticmethod
    def _pass_protocol(msg):
        return isinstance(msg, dict) and progress.prog in msg

    def _update(self):
        try:
            self.p = self.count / self.total
            if self.p > self.p_prev:
                self.p_prev = self.p
        finally:
            pass


class AsyncProgressCounter(AsyncProgressHandler):
    def _done(self):
        print()
        print("done")

    def _update(self):
        super()._update()
        tqdm.write(f'finished {int(self.p * 100)}% of {self.total} interpolations ... \r', end="")


class AsyncProgressBar(AsyncProgressHandler):

    def _init(self, total, step):
        super()._init(total, step)
        self.pbar = tqdm(total=total, desc="finished", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        # self.pbar = Progress()
        # self.task = self.pbar.add_task("finished", total=total)
        # self.pbar.start_task(self.task)
        # self.pbar.start()

    def _done(self):
        self.pbar.close()
        # self.pbar.start_task(self.task)
        # self.pbar.stop()

    def _update(self):
        super()._update()
        self.pbar.update()
        # self.pbar.update(self.task, advance=self.p)
        # self.pbar.refresh()


FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


class level:
    trace = "TRACE"
    debug = "DEBUG"
    info = "INFO"
    warn = "WARNING"
    error = "ERROR"
    fatal = "FATAL"
