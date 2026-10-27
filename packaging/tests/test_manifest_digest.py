"""Installed/core digest contracts, adapted from the reviewed service prototype.

Tiny synthetic fixtures test validation, lifetimes and hash work, not speed or
model accuracy. The implementation under test is the shared Engine loader.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
import rasterio

from aureon.recipe import S2_BANDS, INPUT_UNITS
from aureon.scene_pipeline import job as accepted
from aureon.cloud import single_pass_source as frozen
from aureon.scene_pipeline.stream_io import WindowSource


def measured_load(path, *, accounting=None, **kwargs):
    stats = {} if accounting is None else accounting
    stats.clear(); stats.update(hash_calls=0, hashed_bytes=0)
    original = accepted.file_hash
    def count(member):
        stats["hash_calls"] += 1
        stats["hashed_bytes"] += Path(member).stat().st_size
        return original(member)
    with patch.object(accepted, "file_hash", count):
        return accepted.load_manifest(path, **kwargs)


class Candidate:
    load_manifest = staticmethod(measured_load)

candidate = Candidate()



class PhaseVerifierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='aureon-digest-test-')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.scene = self.folder/'scene'; self.scene.mkdir()
        self.path = self.scene/'manifest.json'
        self.raster = self.scene/'scene.tif'; self.mask = self.scene/'mask.tif'
        transform = rasterio.transform.from_origin(1000,2000,10,10)
        common = dict(driver='GTiff',height=19,width=31,crs='EPSG:32647',transform=transform,
                      tiled=True,blockxsize=16,blockysize=16)
        values = np.arange(12*19*31,dtype=np.float32).reshape(12,19,31)
        with rasterio.open(self.raster,'w',count=12,dtype='float32',**common) as dst:
            dst.write(values); dst.descriptions=S2_BANDS
        valid=np.ones((19,31),np.uint8); valid[0,0]=0
        with rasterio.open(self.mask,'w',count=1,dtype='uint8',**common) as dst: dst.write(valid,1)
        metadata=self.scene/'metadata.xml'; metadata.write_bytes(b'<test>tiny</test>')
        self.record={'schema':'aureon.sentinel2_scene.v1','scene_id':'tiny-synthetic',
            'raster':dict(self.declaration(self.raster),bands=list(S2_BANDS),units=INPUT_UNITS,
                crs='EPSG:32647',shape=[12,19,31],dtype='float32',transform=list(transform)[:6]),
            'valid_mask':dict(self.declaration(self.mask),valid_pixels=int(valid.sum())),
            'source':{'metadata':self.declaration(metadata)}}
        self.save()

    def declaration(self,path):
        return {'path':str(path.relative_to(self.scene)),'sha256':accepted.file_hash(path),'bytes':path.stat().st_size}
    def save(self): self.path.write_text(json.dumps(self.record))
    def reject_both(self,**kwargs):
        for loader in (accepted.load_manifest,):
            with self.subTest(loader=loader.__module__),self.assertRaises((ValueError,KeyError,FileNotFoundError)):
                loader(self.path,**kwargs)
    @contextmanager
    def source(self):
        with frozen.SinglePassSource(self.path) as source: yield source
    def test_equal_contract_and_exact_tiny_read_accounting(self):
        self.record['repeated']=copy.deepcopy(self.record['raster']); self.save()
        calls=[]; original=accepted.file_hash
        def count(path): calls.append(Path(path)); return original(path)
        with patch.object(accepted,'file_hash',count): expected=accepted.load_manifest(self.path)
        stats={}; actual=candidate.load_manifest(self.path,accounting=stats)
        self.assertEqual(actual,expected)
        self.assertEqual(len(calls),3); self.assertEqual(stats['hash_calls'],3)
        self.assertEqual(actual[0],self.record)
        self.assertEqual(actual[1],{'raster':self.raster,'valid_mask':self.mask})
        self.assertEqual(actual[2],hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(stats['hashed_bytes'],sum(p.stat().st_size for p in set(calls)))

    def test_conflicting_repeated_hash_is_rejected(self):
        self.record['repeated']=dict(self.record['raster'],sha256='0'*64); self.save(); self.reject_both()
    def test_conflicting_repeated_size_is_rejected(self):
        self.record['repeated']=dict(self.record['raster'],bytes=1); self.save(); self.reject_both()
    def test_non_integer_byte_expectation_rejected_on_digest_hit(self):
        for invalid in (True,1.0,'1'):
            self.record['repeated']=dict(self.record['raster'],bytes=invalid); self.save(); self.reject_both()
    def test_duplicate_declarations_still_charge_admission_budget(self):
        self.record['repeated']=copy.deepcopy(self.record['raster']); self.save()
        unique=self.raster.stat().st_size+self.mask.stat().st_size+(self.scene/'metadata.xml').stat().st_size
        self.reject_both(max_evidence_bytes=unique)
    def test_declaration_limit_not_expanded(self):
        self.record['repeat']=[copy.deepcopy(self.record['raster']) for _ in range(62)]; self.save(); self.reject_both()
    def test_invalid_limits_unchanged(self):
        for value in (True,0,2*1024**3+1,1.5): self.reject_both(max_evidence_bytes=value)
    def test_cache_is_new_for_each_phase_and_job(self):
        stats={}
        for _ in range(3):
            candidate.load_manifest(self.path,accounting=stats)
            self.assertEqual(stats['hash_calls'],3)
    def test_same_size_same_mtime_raster_mutation_fails_postflight(self):
        # DrvFS may round utime to whole seconds. Establish an exactly
        # representable timestamp before opening the source, not after mutation.
        stamp=1700000000*10**9
        os.utime(self.raster,ns=(stamp,stamp))
        with self.source() as source:
            before=self.raster.stat()
            with self.raster.open('r+b') as stream:
                stream.seek(-1,2); value=stream.read(1); stream.seek(-1,2); stream.write(bytes([value[0]^1]))
            os.utime(self.raster,ns=(before.st_atime_ns,before.st_mtime_ns))
            self.assertEqual(self.raster.stat().st_size,before.st_size)
            self.assertEqual(self.raster.stat().st_mtime_ns,before.st_mtime_ns)
            with self.assertRaisesRegex(ValueError,'hash mismatch'): source.verify()
    def test_mask_mutation_fails_postflight(self):
        with self.source() as source:
            with self.mask.open('ab') as stream: stream.write(b'changed')
            with self.assertRaises(ValueError): source.verify()
    def test_manifest_mutation_fails_postflight(self):
        with self.source() as source:
            self.path.write_bytes(self.path.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'manifest changed'): source.verify()
    def test_metadata_member_mutation_fails_postflight(self):
        with self.source() as source:
            (self.scene/'metadata.xml').write_bytes(b'<test>edit</test>')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): source.verify()
    def test_existing_sidecar_mutation_fails_postflight(self):
        side=Path(str(self.raster)+'.aux.xml')
        xml='<PAMDataset><Metadata><MDI key="test">before</MDI></Metadata></PAMDataset>'
        side.write_text(xml)
        with self.source() as source:
            side.write_text(xml.replace('before','after!'))
            with self.assertRaisesRegex(ValueError,'supporting files changed'): source.verify()
    def test_new_sidecar_fails_postflight(self):
        with self.source() as source:
            Path(str(self.mask)+'.aux.xml').write_text('<PAMDataset><Metadata><MDI key="test">new</MDI></Metadata></PAMDataset>')
            with self.assertRaisesRegex(ValueError,'supporting files changed'): source.verify()
    def test_same_resolved_path_alias_reuses_content_but_not_budget(self):
        (self.scene/'inner').mkdir()
        self.record['alias']=dict(self.record['raster'],path='inner/../scene.tif'); self.save()
        stats={}; actual=candidate.load_manifest(self.path,accounting=stats)
        self.assertEqual(actual[0],self.record); self.assertEqual(stats['hash_calls'],3)
    def test_leaf_symlink_inside_scene_is_rejected(self):
        link=self.scene/'linked.tif'; link.symlink_to(self.raster)
        self.record['alias']=dict(self.record['raster'],path='linked.tif'); self.save(); self.reject_both()
    def test_parent_traversal_outside_scene_is_rejected(self):
        outside=self.folder/'outside.bin'; outside.write_bytes(b'outside')
        self.record['escape']={'path':'../outside.bin','sha256':accepted.file_hash(outside),'bytes':7}; self.save(); self.reject_both()
    def test_directory_symlink_escape_is_rejected(self):
        outside=self.folder/'external'; outside.mkdir(); (outside/'item').write_bytes(b'outside')
        (self.scene/'link').symlink_to(outside,target_is_directory=True)
        self.record['escape']={'path':'link/item','sha256':accepted.file_hash(outside/'item'),'bytes':7}; self.save(); self.reject_both()
    def test_safe_internal_directory_alias_matches_original_policy(self):
        nested=self.scene/'nested'; nested.mkdir(); payload=nested/'item'; payload.write_bytes(b'internal')
        (self.scene/'alias').symlink_to(nested,target_is_directory=True)
        self.record['a']=self.declaration(payload)
        self.record['b']=dict(self.record['a'],path='alias/item'); self.save()
        stats={}; self.assertEqual(candidate.load_manifest(self.path,accounting=stats)[0],self.record)
        self.assertEqual(stats['hash_calls'],4)
    def test_hardlinks_with_distinct_resolved_paths_are_not_coalesced(self):
        link=self.scene/'hardlink.tif'; os.link(self.raster,link)
        self.record['hardlink']=self.declaration(link); self.save()
        stats={}; self.assertEqual(candidate.load_manifest(self.path,accounting=stats)[0],self.record)
        self.assertEqual(stats['hash_calls'],4)
    def test_failure_does_not_seed_a_future_phase(self):
        self.record['raster']['sha256']='0'*64; self.save()
        with self.assertRaises(ValueError): candidate.load_manifest(self.path)
        self.record['raster']['sha256']=accepted.file_hash(self.raster); self.save()
        stats={}; candidate.load_manifest(self.path,accounting=stats); self.assertEqual(stats['hash_calls'],3)
    def test_valid_normalized_groups_unchanged_under_adapter(self):
        with WindowSource(self.path) as source:
            expected=[(values.copy(),windows) for values,windows in source.groups()]; source.verify()
        with self.source() as source:
            actual=[(values.copy(),windows) for values,windows in source.groups()]; source.verify()
        self.assertEqual(len(actual),len(expected))
        for (left,lw),(right,rw) in zip(actual,expected):
            self.assertEqual(lw,rw); np.testing.assert_array_equal(left,right)
