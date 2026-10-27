"""Offline streamed reference reconstruction and complete COG comparison."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .job import file_hash,write_json,canonical_hash
from .stream_io import WindowSource,DiskBlend,CategorySink,windows,rio_window,_grid,MAX_SIDE
from .streaming import ReferenceCursor,_cleanup_owned


def audit_delivery(delivery,reference,output):
    """No GPU; fresh directory holds independent disk reconstruction and report."""
    import rasterio
    delivery,reference,output=map(Path,(delivery,reference,output))
    output.mkdir(exist_ok=False)
    report={'schema':'aureon.streaming_audit.v1','status':'running'}
    try:
        saved=json.loads((delivery/'streaming.json').read_text())
        reference_receipt=json.loads((reference/'reference.json').read_text())
        if saved['status']!='completed' or not saved['parity']['bitwise_equal']:
            raise ValueError('Unpublished or failed streaming delivery')
        if file_hash(reference/'reference.json')!=saved['parity']['reference_sha256']:
            raise ValueError('Reference receipt identity changed')
        if file_hash(delivery/'groups.jsonl')!=saved['groups_sha256']:
            raise ValueError('Candidate ledger changed')
        # Ledger bounded by the admitted max625 tiles, max79 groups (<1MiB).
        if max((delivery/'groups.jsonl').stat().st_size,(reference/'groups.jsonl').stat().st_size)>1024**2:
            raise ValueError('Oversized group ledger')
        if (delivery/'groups.jsonl').read_bytes()!=(reference/'groups.jsonl').read_bytes():
            raise ValueError('Candidate and reference ordered group identities differ')
        if saved['candidate_logits_retained'] and file_hash(delivery/'logits.bin')!=reference_receipt['logits_sha256']:
            raise ValueError('Saved candidate logits differ')
        with WindowSource(reference_receipt['manifest_path']) as source:
            if saved['source_manifest_sha256']!=source.manifest_hash:raise ValueError('Source changed')
            with ExitStack() as stack:
                cursor=stack.enter_context(ReferenceCursor(reference,source,saved['model_identity'],
                    precision_contract=saved.get('precision_contract')))
                blend=stack.enter_context(DiskBlend(output,source))
                sink=stack.enter_context(CategorySink(output/'reconstructed',source))
                count=0
                for inputs,selected in source.groups():
                    logits,_=cursor.next(inputs,selected)
                    for window,values in zip(selected,logits):blend.add(window,values)
                    count+=len(inputs)
                cursor.finish();blend.finish(sink)
                if count!=saved['tile_count'] or source.skipped!=saved['tiles_skipped']:
                    raise ValueError('Tile/skip accounting changed')
            artifacts=json.loads((delivery/'artifacts/receipt.json').read_text())
            if artifacts!=saved['artifacts'] or canonical_hash(artifacts['provenance'])!=artifacts['provenance_sha256']:
                raise ValueError('Artifact provenance identity changed')
            for name,member,nodata in [('class_map.tif','classes.tif',255),('valid_mask.tif','mask.tif',0)]:
                target=delivery/'artifacts'/name
                if file_hash(target)!=artifacts['files'][name]['sha256']:raise ValueError('COG bytes changed')
                with rasterio.open(target) as actual,rasterio.open(output/'reconstructed'/member) as expected:
                    if (_grid(actual,source.grid['crs'],rasterio,max_side=MAX_SIDE)!=source.grid
                            or actual.dtypes!=('uint8',) or actual.nodata!=nodata
                            or actual.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')!='COG'
                            or actual.tags().get('AUREON_PROVENANCE_SHA256')!=artifacts['provenance_sha256']):
                        raise ValueError('COG schema/grid/provenance changed')
                    for window in windows(source.height,source.width):
                        w=rio_window(window)
                        if (not np.array_equal(actual.read(1,window=w),expected.read(1,window=w))
                                or not np.array_equal(actual.read_masks(1,window=w)>0,source.valid(window))):
                            raise ValueError('Reconstructed categorical pixels or masks differ')
            source.verify()
            report.update(status='passed',tile_count=count,skipped=source.skipped,
                normalization_and_order_recomputed=True,disk_blend_reconstructed=True,
                cog_pixels_masks_grid_provenance_verified=True,
                retained_candidate_logits_verified=saved['candidate_logits_retained'],
                source_manifest_sha256=source.manifest_hash,
                scope='CPU-only complete disk-reference reconstruction; candidate execution bound by group hashes and optional saved logits')
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        _cleanup_owned(output,('accumulator.tif','coverage.tif'))
        write_json(output/'audit.json',report)
    return report


def audit_streaming_job(folder,output):
    folder=Path(folder);output=Path(output)
    job=json.loads((folder/'job.json').read_text())
    if job.get('schema')!='aureon.streaming_scene_job.v1' or job.get('status')!='completed':
        raise ValueError('Expected completed streaming job')
    startup=json.loads((folder/'worker-startup.json').read_text())
    reference=json.loads((folder/'reference/reference.json').read_text())
    delivery=json.loads((folder/'delivery/streaming.json').read_text())
    identity=reference['model_identity']
    precision=job['precision']
    if (startup['status']!='ready' or startup['model_id']!=job['model_id'] or identity!=delivery['model_identity']
            or identity['model_id']!=job['model_id'] or identity['precision']!=precision
            or startup['precision']!=precision or precision['float32_matmul_precision']!='highest'
            or precision['matmul_allow_tf32'] or precision['cudnn_allow_tf32']
            or identity['bundle_receipt_sha256']!=job['bundle_receipt_sha256']
            or identity['bundle_receipt_sha256']!=startup['bundle_receipt_sha256']
            or identity['bundle_files_sha256']!=startup['bundle_files_sha256']
            or file_hash(folder/'worker-startup.json')!=job['worker']['worker_startup_receipt_sha256']
            or startup['worker_id']!=job['worker']['worker_id']
            or file_hash(folder/'input-manifest.json')!=job['source_manifest_sha256']
            or job['source_manifest_sha256']!=reference['source_manifest_sha256']
            or job['tile_count']!=delivery['tile_count'] or job['parity']!=delivery['parity']):
        raise ValueError('Worker/model/input/precision linkage changed')
    for name,digest in job['files_sha256'].items():
        path=folder/name
        if not path.resolve().is_relative_to(folder.resolve()) or file_hash(path)!=digest:
            raise ValueError('Streaming job member identity changed')
    for name,digest in job['source_sha256'].items():
        source=folder/'source-snapshot'/name
        if (not source.resolve().is_relative_to((folder/'source-snapshot').resolve())
                or file_hash(source)!=digest):
            raise ValueError('Captured runtime source identity changed')
    result=audit_delivery(folder/'delivery',folder/'reference',output.parent/(output.stem+'-reconstruction'))
    result={**result,'job_id':job['job_id'],'model_name':job['model_name'],'model_identity_checked':True}
    write_json(output,result)
    return result
