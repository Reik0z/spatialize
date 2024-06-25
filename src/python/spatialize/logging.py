import json
import logging
import os

from rich.logging import RichHandler
from tqdm.auto import tqdm

FORMAT = "%(message)s"
logging.basicConfig(
    level="ERROR",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_time=False, show_path=False)]
)

log = logging.getLogger("rich")


# ************************************ PROTOCOL ************************************
# Messages for logging:
#
#   {"message": {"text": "<the log text>", "level": "<DEBUG|INFO|WARNING|ERROR|CRITICAL>"}}
#
# Messages for showing progress:
#
#  1. To start a new progress counting:
#   {"progress": {"init": <total expected count>, "step": <increment step>}}
#
#  2. To inform during a progress counting
#   {"progress": {"token": <value>}}
#
#  3. To finish a progress counting
#   {"progress": "done"}
# **********************************************************************************
class level:
    debug = "DEBUG"
    info = "INFO"
    warn = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"


class logger:
    message = "message"
    text = "text"
    level = "level"

    @classmethod
    def debug(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.debug}}

    @classmethod
    def info(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.info}}

    @classmethod
    def warning(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.warn}}

    @classmethod
    def error(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.error}}

    @classmethod
    def critical(cls, msg):
        return {cls.message: {cls.text: msg, cls.level: level.critical}}


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


# **************************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ********************************* HANDLERS ***************************************
class MessageHandler:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, msg):
        for callback in self.callbacks:
            callback(msg)


class LogMessage:

    def __init__(self):
        pass

    def __call__(self, msg):
        global log

        if not self._pass_protocol(msg):
            return

        lev = msg[logger.message][logger.level]
        text = msg[logger.message][logger.text]

        if lev == level.debug:
            log.debug(text)

        if lev == level.info:
            log.info(text)

        if lev == level.warn:
            log.warning(text)

        if lev == level.error:
            log.error(text)

        if lev == level.critical:
            log.critical(text)

    @staticmethod
    def _pass_protocol(msg):
        return isinstance(msg, dict) and logger.message in msg


log_message = LogMessage()  # to use it in the plain python code


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
        try:
            m = self._pass_protocol(msg)
        except:
            return
        # print(type(m), m)
        if isinstance(m[progress.prog], dict):
            if progress.init in m[progress.prog]:
                print(m)
                self._init(m[progress.prog][progress.init], m[progress.prog][progress.step])
            if progress.token in m[progress.prog]:
                # no need to process the incoming value
                self._increment()
                self._update()
        if m[progress.prog] == progress.done:
            self._done()
            self._reset()

    def _init(self, total, step):
        self.total = total
        self.step = step

    def _reset(self):
        self.count = 0
        self.p = 0
        self.p_prev = -1
        self.ready_to_update = False

    def _increment(self):
        self.count += self.step

    @staticmethod
    def _pass_protocol(msg):
        m = msg
        if isinstance(msg, str):
            m = json.loads(msg)
        if isinstance(m, dict) and progress.prog in m:
            return m
        raise TypeError

    def _update(self):
        try:
            self.ready_to_update = False
            self.p = (self.count * 100) // self.total
            if self.p > self.p_prev:
                self.p_prev = self.p
                self.ready_to_update = True
        finally:
            pass


class AsyncProgressCounter(AsyncProgressHandler):
    def _done(self):
        tqdm.write(os.linesep)
        tqdm.write("done")

    def _update(self):
        super()._update()
        if self.ready_to_update:
            tqdm.write(f'finished {int(self.p)}% of {self.total} iterations ... \r', end="")


class AsyncProgressBar(AsyncProgressHandler):

    def _init(self, total, step):
        super()._init(total, step)
        self.pbar = tqdm(total=total, desc="finished", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    def _done(self):
        self.pbar.close()

    def _update(self):
        super()._update()
        self.pbar.update()
# **************************++++++++++++++++++++++++++++++++++++++++++++++++++++++++
