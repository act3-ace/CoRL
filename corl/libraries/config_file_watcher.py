"""

Confi File Watcher Module
"""
import logging
import logging.config
import os
import pathlib
import pprint
from datetime import datetime
from logging import FileHandler
from logging.handlers import RotatingFileHandler

import yaml
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

FILE_PATH = str(pathlib.Path(__file__).parent.absolute())


class PrettyLog:
    """Print util for dicts using logger"""

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return pprint.pformat(self.obj)


class PrettyLogNL:
    """Print util for dicts using logger"""

    def __init__(self, obj):
        self.obj = obj

    def __repr__(self):
        return "\n" + pprint.pformat(self.obj)


class LoggingSetup:
    """
    Logging Setup Config
    """

    def __init__(
        self,
        default_path: str = os.path.join(os.getcwd(), "logging.yml"),
        default_level=logging.INFO,
        env_key: str = "LOG_CFG",
        enabled_file_watch=False,
    ):
        if enabled_file_watch:
            self._create_event_handler()
            self._setup_event_handler()
            self._create_observers()
        self._default_path = default_path
        self._default_level = default_level
        self._env_key = env_key
        self._setup_logging(self._default_path, self._default_level)
        self._log = logging.getLogger(LoggingSetup.__name__)

    def _setup_logging(self, default_path, default_level):
        self._log_file = default_path
        if os.path.exists(self._log_file):
            with open(self._log_file, encoding="utf8") as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)

    def _create_event_handler(self):
        self._patterns = "*"
        self._ignore_patterns = ""
        self._ignore_directories = False
        self._case_sensitive = True
        self._event_handler = PatternMatchingEventHandler(
            self._patterns,
            self._ignore_patterns,
            self._ignore_directories,
            self._case_sensitive,
        )

    def _setup_event_handler(self):
        self._event_handler.on_created = self._on_created
        self._event_handler.on_deleted = self._on_deleted
        self._event_handler.on_modified = self._on_modified
        self._event_handler.on_moved = self._on_moved

    def _create_observers(self):
        self._path = "."
        self._go_recursively = True
        self._observer = Observer()
        self._observer.schedule(self._event_handler, self._path, recursive=self._go_recursively)
        self._observer.start()

    def _on_created(self, event):
        self._log.debug(f"{event.src_path} has been created!")

    def _on_deleted(self, event):
        self._log.debug(f"{event.src_path} has been deleted!")

    def _on_modified(self, event):
        self._log.debug(f"{event.src_path} has been modified!")
        if self._log_file in event.src_path:
            self._setup_logging(self._default_path, self._default_level, self._env_key)

    def _on_moved(self, event):
        self._log.debug(f"{event.src_path} has been moved!")


class TimestampedFileHandler(FileHandler):
    """FileHandler that adds timestamp to log filename"""

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        FileHandler.__init__(
            self,
            TimestampedFileHandler.timestamp_filename(filename),
            mode,
            encoding,
            delay,
        )

    @staticmethod
    def timestamp_filename(base_filename: str) -> str:
        """insert timestamp into filename as a prefix

        Parameters
        ----------
        base_filename : str
            the filename to insert timestamp into

        Returns
        -------
        str
            the filename with timestamp inserted
        """
        path, filename = os.path.split(base_filename)
        t = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%fZ")
        filename = f"{t}_{filename}"
        return os.path.join(path, filename)


class TimestampedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that adds timestamp to log filename"""

    def __init__(self, filename, mode="a", maxBytes=0, backupCount=0, encoding=None, delay=False):
        RotatingFileHandler.__init__(
            self,
            TimestampedFileHandler.timestamp_filename(filename),
            mode,
            maxBytes,
            backupCount,
            encoding,
            delay,
        )
        self._custom_base_filename = filename

    @staticmethod
    def _rotation_filename(default_name, postfix=""):
        filename = TimestampedFileHandler.timestamp_filename(default_name)
        filename, ext = os.path.splitext(filename)
        return filename + postfix + ext

    def doRollover(self, postfix=""):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        self.baseFilename = self._rotation_filename(self._custom_base_filename, postfix)
        if not self.delay:
            self.stream = self._open()
