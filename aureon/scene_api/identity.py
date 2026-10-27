"""Cheap local lifecycle guards after full cryptographic startup verification.

These guards detect ordinary file/tensor mutation. They are not new value hashes
or a security boundary against another process with the same filesystem powers.
"""
from pathlib import Path
import stat


class WorkerIntegrityError(RuntimeError):
    pass


def file_stamp(path):
    path = Path(path)
    value = path.lstat()
    # lstat describes the link itself. S_ISREG therefore rejects symlinks and
    # other nonregular members without a second metadata lookup/race window.
    if not stat.S_ISREG(value.st_mode):
        raise WorkerIntegrityError('Worker identity member must remain a regular file')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


class FileSeal:
    def __init__(self, paths):
        if not paths:
            raise ValueError('Explicit identity members required')
        self._files = {str(Path(path).absolute()): file_stamp(path) for path in paths}

    def verify(self):
        for path, expected in self._files.items():
            try:
                actual = file_stamp(path)
            except OSError as error:
                raise WorkerIntegrityError('Worker identity member disappeared') from error
            if actual != expected:
                raise WorkerIntegrityError('Worker identity member changed: '+path)
        return len(self._files)

    def records(self):
        return {path: list(stamp) for path, stamp in self._files.items()}


class TensorSeal:
    def __init__(self, model):
        self._model = model
        self._state = self._describe()

    def _describe(self):
        if any(module.training for module in self._model.modules()):
            raise WorkerIntegrityError('Every resident model module must remain in evaluation mode')
        return {name: (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype),
                       str(tensor.device), tensor._version, tensor.requires_grad)
                for name, tensor in self._model.state_dict(keep_vars=True).items()}

    def verify(self):
        if self._describe() != self._state:
            raise WorkerIntegrityError('Resident tensor identity/version changed')
        return len(self._state)
