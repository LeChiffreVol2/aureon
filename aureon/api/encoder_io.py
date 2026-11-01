"""Explicit local feature export for the TerraMind encoder CLI; no pickle files."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, IO, Protocol, TypeAlias, cast
import ast
from itertools import product
import struct
import traceback
import zipfile

from .v1 import ExecutionRequired
from ..terramind_native.common import binding, write_json
from ..terramind_native.encoder_loader import CHANNELS, MergeMethod

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from torch import Tensor
    from ..terramind_native.encoder_loader import FeatureLayer
    from .terramind_encoder import TerraMindEncoder, TerraMindFeatures

EncoderArray: TypeAlias = 'NDArray[np.float32] | NDArray[np.int64]'


_INPUT_LIMIT = 512 * 1024**2
_HEADER_LIMIT = 10_000
_DIRECTORY_LIMIT = 64 * 1024
_READ_CHUNK = 1024 * 1024


class _ReadInto(Protocol):
    def readinto(self, buffer: memoryview, /) -> int | None: ...


@dataclass(frozen=True)
class _ArrayHeader:
    modality: str
    shape: tuple[int, ...]
    fortran_order: bool
    dtype: str
    payload_bytes: int
    payload_offset: int


def _bounded_archive(source: Path) -> None:
    """Bound central-directory allocation before constructing ``ZipFile``.

    Supports standard single-disk NPZ with stored/deflated numeric members and
    bounded comments/extra fields. ZIP64 end records and concatenated archives
    are rejected. NumPy's small local ZIP64 member headers remain supported.
    """
    with source.open('rb') as stream:
        stream.seek(0,2)
        size=stream.tell()
        stream.seek(max(0,size-(65535+22)))
        tail=stream.read(65535+22)
        offset=tail.rfind(b'PK\x05\x06')
        if offset<0 or len(tail)-offset<22:
            raise ValueError('NPZ requires a bounded standard ZIP end record')
        _,disk,start_disk,on_disk,entries,directory_bytes,directory_offset,comment_bytes=struct.unpack_from('<4s4H2IH',tail,offset)
        end_offset=size-len(tail)+offset
        if (offset+22+comment_bytes!=len(tail) or disk or start_disk or on_disk!=entries
                or not 1<=entries<=len(CHANNELS) or not 46<=directory_bytes<=_DIRECTORY_LIMIT
                or directory_offset+directory_bytes!=end_offset):
            raise ValueError('NPZ must be single-disk with at most eight members and a bounded non-ZIP64 central directory')
        stream.seek(directory_offset)
        directory=stream.read(directory_bytes)
    # Count actual records, not merely the untrusted EOCD count. ZipFile reads
    # until size_cd is exhausted regardless of its declared entry count.
    position=count=0
    while position<len(directory):
        if count>=len(CHANNELS) or len(directory)-position<46 or directory[position:position+4]!=b'PK\x01\x02':
            raise ValueError('Invalid or oversized NPZ central directory')
        flags,compression=struct.unpack_from('<HH',directory,position+8)
        name_bytes,extra_bytes,member_comment=struct.unpack_from('<HHH',directory,position+28)
        disk_start=struct.unpack_from('<H',directory,position+34)[0]
        if flags&1 or compression not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) or disk_start:
            raise ValueError('NPZ requires unencrypted stored/deflated members on one disk')
        position+=46+name_bytes+extra_bytes+member_comment
        count+=1
    if position!=directory_bytes or count!=entries:
        raise ValueError('NPZ central directory length/count mismatch')


def _member_header(stream: IO[bytes], entry: zipfile.ZipInfo) -> _ArrayHeader:
    """Validate NPY metadata before any shape-dependent array allocation."""
    import numpy as np

    modality = entry.filename[:-4]
    if modality not in CHANNELS:
        raise ValueError('NPZ members must use canonical supported modality names')
    prefix = stream.read(8)
    if len(prefix) != 8 or prefix[:6] != b'\x93NUMPY':
        raise ValueError('Malformed NPY magic/version')
    version = tuple(prefix[6:])
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError('Unsupported NPY version')
    length_size = 2 if version == (1, 0) else 4
    length = stream.read(length_size)
    if len(length) != length_size:
        raise ValueError('Truncated NPY header length')
    header_bytes = int.from_bytes(length, 'little')
    if not 1 <= header_bytes <= _HEADER_LIMIT:
        raise ValueError('NPY header exceeds its bounded size')
    encoded = stream.read(header_bytes)
    if len(encoded) != header_bytes:
        raise ValueError('Truncated NPY header')
    try:
        header = ast.literal_eval(encoded.decode('utf-8' if version == (3, 0) else 'latin1'))
    except (ValueError, SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError('Malformed NPY header') from error
    if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'}:
        raise ValueError('Malformed NPY header fields')
    shape, order, descriptor = header['shape'], header['fortran_order'], header['descr']
    tail = (224, 224) if modality == 'LULC' else (CHANNELS[modality], 224, 224)
    if (not isinstance(shape, tuple) or len(shape) != len(tail) + 1
            or any(type(size) is not int or size < 1 for size in shape)
            or shape[1:] != tail or type(order) is not bool or not isinstance(descriptor, str)):
        raise ValueError('NPY must declare a positive full-band 224x224 batch and a boolean order')
    try:
        dtype = np.dtype(descriptor)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid NPY dtype') from error
    expected = np.dtype('int64' if modality == 'LULC' else 'float32')
    if dtype != expected or not dtype.isnative:
        raise ValueError('NPY dtype must be native int64 for LULC or native FP32 for continuous inputs')
    # Python integer division avoids fixed-width overflow and never constructs an
    # oversized product, including headers with adversarially large dimensions.
    payload_bytes = dtype.itemsize
    for size in shape:
        if size > _INPUT_LIMIT // payload_bytes:
            raise ValueError('NPY declared array exceeds 512 MiB')
        payload_bytes *= size
    offset = stream.tell()
    if entry.file_size != offset + payload_bytes:
        raise ValueError('NPY payload size differs from its declared shape/dtype')
    return _ArrayHeader(modality, shape, order, expected.str, payload_bytes, offset)


def _archive_headers(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, _ArrayHeader]]:
    entries = archive.infolist()
    if (not 1 <= len(entries) <= len(CHANNELS) or sum(e.file_size for e in entries) > _INPUT_LIMIT
            or len({e.filename for e in entries}) != len(entries)
            or any(not e.filename.endswith('.npy') or '/' in e.filename or '\\' in e.filename for e in entries)):
        raise ValueError('Expected a bounded NPZ containing unique modality arrays')
    headers = []
    batch: int | None = None
    for entry in entries:
        with archive.open(entry) as stream:
            header = _member_header(stream, entry)
        if batch is not None and batch != header.shape[0]:
            raise ValueError('NPY input batch sizes must match')
        batch = header.shape[0]
        headers.append((entry, header))
    return headers


def _read_exact(stream: IO[bytes], view: memoryview) -> None:
    offset=0
    while offset<len(view):
        chunk=view[offset:min(offset+_READ_CHUNK,len(view))]
        try:
            # ZipFile.open is typed as IO[bytes], but its ZipExtFile implements
            # the binary buffered readinto interface used here.
            read=cast(_ReadInto,stream).readinto(chunk)
        finally:
            chunk.release()
        if not read:
            raise ValueError('Truncated NPY payload')
        offset+=read


def _read_payload(stream: IO[bytes], array: EncoderArray, header: _ArrayHeader) -> None:
    """Fill one C-contiguous destination with at most a 1 MiB decoding buffer."""
    import numpy as np
    if not header.fortran_order:
        view=memoryview(array).cast('B')
        try:
            _read_exact(stream,view)
        finally:
            view.release()
    else:
        # Consume contiguous F-order slabs and assign their logical slices into
        # the C-order destination. No whole-array F-to-C duplicate is needed.
        axis=0
        prefix_bytes=array.dtype.itemsize
        while axis<len(header.shape) and prefix_bytes*header.shape[axis]<=_READ_CHUNK:
            prefix_bytes*=header.shape[axis]
            axis+=1
        storage=bytearray(min(_READ_CHUNK,header.payload_bytes))
        view=memoryview(storage)
        try:
            if axis==len(header.shape):
                _read_exact(stream,view)
                array[...] = np.frombuffer(view,dtype=array.dtype).reshape(header.shape,order='F')
            else:
                step=max(1,_READ_CHUNK//prefix_bytes)
                for reversed_tail in product(*(range(size) for size in reversed(header.shape[axis+1:]))):
                    tail=tuple(reversed(reversed_tail))
                    for start in range(0,header.shape[axis],step):
                        count=min(step,header.shape[axis]-start)
                        chunk=view[:count*prefix_bytes]
                        try:
                            _read_exact(stream,chunk)
                            selection=(slice(None),)*axis+(slice(start,start+count),)+tail
                            array[selection] = np.frombuffer(chunk,dtype=array.dtype).reshape(header.shape[:axis]+(count,),order='F')
                        finally:
                            chunk.release()
        finally:
            view.release()
    if stream.read(1):
        raise ValueError('Oversized NPY payload')


def _validate_values(array: EncoderArray, modality: str) -> None:
    """Bound numeric validation temporaries as well as decompression buffers."""
    import numpy as np
    flat=array.reshape(-1)
    step=max(1,_READ_CHUNK//array.dtype.itemsize)
    for start in range(0,flat.size,step):
        chunk=flat[start:start+step]
        if modality=='LULC':
            if bool((chunk<0).any()) or bool((chunk>9).any()):
                raise ValueError('LULC must contain class indices 0..9')
        elif not bool(np.isfinite(chunk).all()):
            raise ValueError('Continuous encoder inputs must be finite')


def _read_arrays(input_file: str | Path, modalities: Sequence[str] | None = None) -> tuple[dict[str, EncoderArray], dict[str, object]]:
    """Load bounded, numeric-only NPZ data without importing Torch."""
    import numpy as np

    source = Path(input_file).resolve(strict=True)
    if source.stat().st_size > _INPUT_LIMIT:
        raise ValueError('Feature input exceeds 512 MiB')
    identity = binding(source)
    _bounded_archive(source)
    arrays: dict[str, EncoderArray] = {}
    array: EncoderArray | None = None
    try:
        with zipfile.ZipFile(source) as archive:
            headers = _archive_headers(archive)
            if modalities is not None and (isinstance(modalities, (str, bytes))
                    or len(set(modalities)) != len(modalities)
                    or {h.modality for _, h in headers} != set(modalities)):
                raise ValueError('NPZ modalities must exactly match the configured encoder')
            for entry, header in headers:
                with archive.open(entry) as stream:
                    if _member_header(stream, entry) != header:
                        raise ValueError('Input archive header changed while reading')
                    array = np.empty(header.shape,dtype=np.dtype(header.dtype))
                    _read_payload(stream,array,header)
                _validate_values(array,header.modality)
                arrays[header.modality] = array
                array=None
        if binding(source) != identity:
            raise ValueError('Input archive changed while reading')
        return arrays, identity
    except BaseException as error:
        arrays.clear();array=None
        # Completed decoder/helper frames may retain a buffer passed to readinto.
        # Retaining the exception must not retain already-decoded request arrays.
        traceback.clear_frames(error.__traceback__)
        raise


def inspect_encoder_input(input_file: str | Path, *, modalities: Sequence[str] | None = None) -> dict[str, object]:
    """Validate complete NPZ payloads on CPU, with no Torch/model import.

    The existing 512 MiB aggregate input cap also applies here; decompression
    copies at most 1 MiB per read into preallocated numeric arrays. Standard
    stored/deflated single-disk NPZ is supported (no ZIP64 end records). This validates
    structure and numeric values, not upstream standardization or EO quality.
    """
    arrays, identity = _read_arrays(input_file, modalities)
    try:
        return {'schema': 'aureon.encoder_input_inspection.v1', 'input': identity,
                'modalities': list(arrays), 'batch': next(iter(arrays.values())).shape[0],
                'array_bytes': sum(a.nbytes for a in arrays.values()),
                'arrays': {k: {'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': a.nbytes}
                           for k, a in arrays.items()}, 'model_imported': False,
                'preprocessing': 'caller supplies upstream-standardized values; no transform applied'}
    finally:
        arrays.clear()


def _destination(output: str | Path, max_output_bytes: int) -> Path:
    if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 16 * 1024**3:
        raise ValueError('max_output_bytes must be a positive integer at most 16 GiB')
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError('Feature output must be a new directory under an existing parent')
    return destination


def _export_features(result: TerraMindFeatures, *, model_id: str, identity: dict[str, object],
                     destination: Path, max_output_bytes: int, ownership: str) -> dict[str, object]:
    import numpy as np
    from .lifecycle import close_preserving_error

    arrays = {}
    layout: list[dict[str, object]] = []
    layer: FeatureLayer | None = None
    tensor: Tensor | None = None
    tensors: dict[str, Tensor] = {}
    created = False
    try:
        total = count = 0
        for layer in result['features']:
            tensors = layer if isinstance(layer, dict) else {'merged': layer}
            for tensor in tensors.values():
                total += tensor.numel() * tensor.element_size()
                count += 1
        # Budget includes ZIP/NPY headers and a bounded provenance allowance.
        if total + count * 1024 + 1024**2 > max_output_bytes:
            raise ValueError('Feature output exceeds its declared byte budget')
        for index, layer in enumerate(result['features']):
            tensors = layer if isinstance(layer, dict) else {'merged': layer}
            for modality, tensor in tensors.items():
                key = f'layer_{index:03d}_{modality}'
                if not key.replace('_', '').isalnum():
                    raise ValueError('Invalid feature output modality')
                arrays[key] = tensor.detach().cpu().numpy()
                layout.append({'key': key, 'layer': index, 'modality': modality, 'shape': list(tensor.shape)})
        # Feature tensors are request-owned; release their containers before
        # filesystem export while the CPU arrays retain only required storage.
        result['features'].clear()
        tensors.clear()
        tensor = layer = None
        destination.mkdir(exist_ok=False)
        created = True
        with (destination/'features.npz').open('xb') as stream:
            np.savez(stream, allow_pickle=False, **arrays)
        arrays.clear()
        provenance = {'schema': 'aureon.terramind_feature_export.v1', 'model_id': model_id,
            'input': identity, 'native': result['provenance'], 'layout': layout,
            'features': binding(destination/'features.npz'), 'engine_ownership': ownership,
            'output_semantics': 'latent features; no task head or COG/grid claim'}
        write_json(destination/'provenance.json', provenance)
        if sum(p.stat().st_size for p in destination.iterdir()) > max_output_bytes:
            raise ValueError('Saved feature output exceeds its declared byte budget')
        return {'schema': 'aureon.terramind_feature_result.v1', 'status': 'completed',
            'model_id': model_id, 'features': binding(destination/'features.npz'),
            'provenance': binding(destination/'provenance.json'), 'engine_ownership': ownership}
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        if created:
            close_preserving_error(lambda: write_json(destination/'failure.json',
                {'status': 'failed', 'error_type': type(error).__name__}), error,
                context='Feature failure receipt could not be written')
        raise
    finally:
        result['features'].clear()
        tensors.clear()
        tensor = layer = None
        arrays.clear()


def encode_file_with_engine(engine: TerraMindEncoder, *, input_file: str | Path,
                            output: str | Path, seed: int = 17, execute_model: bool = False,
                            max_output_bytes: int = 2 * 1024**3) -> dict[str, object]:
    """Use a caller-owned resident encoder; never close or reconfigure it.

    The encoder's existing lock fences concurrent predict/close. Validation and
    file export own request buffers only. A model failure retains the encoder's
    own failed-state semantics; the caller must close/recreate that instance.
    """
    if execute_model is not True:
        raise ExecutionRequired('Pass execute_model=True explicitly; no files or models opened')
    from .terramind_encoder import TerraMindEncoder
    if not isinstance(engine, TerraMindEncoder):
        raise TypeError('engine must be TerraMindEncoder')
    if engine.state in ('failed', 'closed'):
        raise RuntimeError('TerraMindEncoder is ' + engine.state)
    destination = _destination(output, max_output_bytes)
    configuration = engine.configuration()
    options = cast(dict[str, object], configuration['options'])
    arrays, identity = _read_arrays(input_file, cast(Sequence[str], options['modalities']))
    import torch
    inputs: dict[str, Tensor] = {}
    try:
        inputs = {name: torch.from_numpy(array) for name, array in arrays.items()}
        result = engine.predict(inputs, seed=seed)
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        inputs.clear()
        arrays.clear()
    return _export_features(result, model_id=engine.model_id, identity=identity,
        destination=destination, max_output_bytes=max_output_bytes, ownership='caller')


def encode_file(*, model_id: str, model_file: str | Path, input_file: str | Path,
                output: str | Path, tokenizer_files: Mapping[str, str | Path] | None = None,
                tim_modalities: Sequence[str] | None = None, device: str = "cpu",
                merge_method: MergeMethod = "mean", seed: int = 17,
                execute_model: bool = False, max_output_bytes: int = 2 * 1024**3) -> dict[str, object]:
    """Own one encoder for an explicit local feature job; no implicit transforms.

    Use encode_file_with_engine to reuse a caller-owned resident session. Both
    paths share bounded numeric input validation and identical feature export.
    """
    if execute_model is not True:
        raise ExecutionRequired("Pass execute_model=True explicitly; no files or models opened")
    destination = _destination(output, max_output_bytes)
    arrays, identity = _read_arrays(input_file)
    import torch
    from .terramind_encoder import TerraMindEncoder
    inputs: dict[str, Tensor] = {}
    try:
        inputs = {name: torch.from_numpy(array) for name, array in arrays.items()}
        with TerraMindEncoder(model_id, model_file=model_file, modalities=tuple(inputs),
                              tokenizer_files=tokenizer_files, tim_modalities=tim_modalities,
                              device=device, merge_method=merge_method, execute_model=True) as engine:
            result = engine.predict(inputs, seed=seed)
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        inputs.clear()
        arrays.clear()
    return _export_features(result, model_id=model_id, identity=identity,
        destination=destination, max_output_bytes=max_output_bytes, ownership='engine_owned_closed')
