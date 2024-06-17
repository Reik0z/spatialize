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


class AsyncProgressBar:
    def __init__(self):
        self.total = None
        self.step = None
        self._reset()

    def __call__(self, msg):
        if not self._pass_protocol(msg):
            return
        if isinstance(msg[progress.prog], dict):
            if progress.init in msg[progress.prog]:
                self._init(msg[progress.prog][progress.init], msg[progress.prog][progress.step])
            if progress.token in msg[progress.prog]:
                # no need to process the incoming value
                self._increment()
                self._print()
        if msg[progress.prog] == progress.done:
            self._reset()

    def _init(self, total, step):
        self.total = total
        self.step = step

    def _reset(self):
        self.count = 0
        self.p_prev = -1

        print()
        print("done")

    def _increment(self):
        self.count += self.step

    def _print(self):
        try:
            p = (self.count * 100) // self.total
            if p > self.p_prev:
                print(f'processing {p}% out of {self.total} ...\r', end="")
                self.p_prev = p
        finally:
            pass


    @staticmethod
    def _pass_protocol(msg):
        return isinstance(msg, dict) and progress.prog in msg


class logger:
    trace = "t"
    debug = "d"
    info = "i"
    warn = "w"
    error = "e"
    fatal = "f"

    def log(self, level, msg):
        pass
