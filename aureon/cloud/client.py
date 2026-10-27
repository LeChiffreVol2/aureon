"""Cloud local-v1 client with explicitly bound guarded profile receipts."""
import argparse
import copy
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from ..client import SceneClient, ClientError, APIError, _decode, _require, _job_id
from .control import CONTRACT, MODEL, file_hash


class CloudClient(SceneClient):
    def status_snapshot(self):
        return self.json('GET','/v1/status')

    def health(self):
        """Return the bounded control-plane health body, including degraded 503."""
        connection_type=http.client.HTTPSConnection if self._scheme=='https' else http.client.HTTPConnection
        connection=connection_type(self._host,self._port,timeout=self.timeout)
        try:
            connection.request('GET','/v1/health',headers={'Authorization':'Bearer '+self._token,'Connection':'close'})
            response=connection.getresponse(); value=_decode(self._read(response,65536))
            if response.status not in (200,503) or value.get('schema')!='aureon.cloud.health.v1':
                error=value.get('error',{})
                raise APIError(response.status,str(error.get('code','invalid_health_response')),
                               'Control-plane health unavailable',response.getheader('Retry-After'))
            _require((response.status==200)==(value['control_plane']['ready'] is True),'Health status/body mismatch')
            return value
        finally:
            connection.close()

    def cancel_job(self, identifier):
        return self.json('POST','/v1/jobs/'+_job_id(identifier)+'/cancel',payload={})

    def json(self, method, path, **kwargs):
        with self._response(method,path,**kwargs) as response:
            return _decode(self._read(response,1024**2))

    def upload_scene(self, spec_path):
        from ..prepare.local import load_spec, local_file
        spec, _, base = load_spec(Path(spec_path))
        spec = copy.deepcopy(spec)
        spec.pop('scene_id',None)
        paths = {}
        def add(value, suffix):
            path = local_file(value,base)
            name = 'file'+str(len(paths))+suffix
            paths[name] = path
            return name
        if spec['kind']=='aligned':
            for key in ('raster','valid_mask'):
                spec[key] = add(spec[key],'.tif')
        else:
            for key in spec['bands']:
                spec['bands'][key] = add(spec['bands'][key],'.tif')
            spec['scl'] = add(spec['scl'],'.tif')
            if 'product_metadata' in spec['radiometry']:
                spec['radiometry']['product_metadata'] = add(spec['radiometry']['product_metadata'],'.xml')
        declaration = {'spec':spec,'files':[{'name':name,'bytes':path.stat().st_size,'sha256':file_hash(path)} for name,path in paths.items()]}
        result = self.json('POST','/v1/inputs',payload=declaration)
        for item in result['uploads']:
            path = paths[item['name']]
            _require(item['url'].startswith('/v1/inputs/'+result['id']+'/files/'+item['name']+'?ticket='),'Invalid upload URL')
            connection_type = http.client.HTTPSConnection if self._scheme=='https' else http.client.HTTPConnection
            connection = connection_type(self._host,self._port,timeout=self.timeout)
            try:
                with path.open('rb') as stream:
                    connection.request('PUT',item['url'],body=stream,headers={'Authorization':'Bearer '+self._token,
                        'Content-Type':'application/octet-stream','Content-Length':str(item['bytes']),'Connection':'close'})
                    response = connection.getresponse()
                    uploaded = _decode(self._read(response,1024**2))
                    _require(response.status==200 and uploaded.get('sha256')==item['sha256'], 'Upload failed; input was not completed')
            finally:
                connection.close()
        self.json('POST','/v1/inputs/'+result['id']+'/complete',payload={})
        return result['id']

    def wait_input(self, identifier, timeout=300):
        return self._wait('/v1/inputs/'+identifier,'ready',timeout)

    def wait_job(self, identifier, timeout=600):
        return self._wait('/v1/jobs/'+identifier,'completed',timeout)

    def _wait(self,path,success,timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            value = self.json('GET',path)
            if value['state']==success:
                return value
            if value['state'] in ('failed','rejected','cancelled','expired'):
                raise ClientError('Operation ended with state '+value['state'])
            time.sleep(.1)
        raise TimeoutError('Operation still pending; retain input/job ID and idempotency key')

    def submit_input(self, identifier, profile, key):
        return self.json('POST','/v1/jobs',payload={'input_id':identifier,'model_id':MODEL,'profile_id':profile},key=key)

    def download_job(self, identifier, output, *, expected_manifest_sha256, expected_qualification_sha256,
                     expected_profile_id=None):
        output = Path(output).absolute()
        _require(not output.exists() and output.parent.is_dir(),'Download directory must be new')
        listing = self.json('GET','/v1/jobs/'+identifier+'/artifacts')
        names = {'class_map.tif','valid_mask.tif','provenance.json','job.json'}
        _require(listing['job_id']==identifier and set(listing['artifacts'])==names,'Expected a real four-artifact result')
        output.mkdir()
        for name,item in listing['artifacts'].items():
            _require(item['url'].startswith('/v1/jobs/'+identifier+'/artifacts/'+name+'?ticket='),'Invalid artifact URL')
            with self._response('GET',item['url']) as response:
                data = self._read(response,CONTRACT['max_artifact_bytes'])
            _require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'Artifact integrity failed')
            with (output/name).open('xb') as stream:
                stream.write(data)
        job, provenance = (_decode((output/name).read_bytes()) for name in ('job.json','provenance.json'))
        _require(job['schema']=='aureon.guarded_scene_job.v2' and job['status']=='completed'
                 and job['job_id']==identifier and job['source_manifest_sha256']==expected_manifest_sha256
                 and job.get('profile_id') in ('fp32-guarded-v2','fp32-stream-guarded-v3')
                 and (expected_profile_id is None or job['profile_id']==expected_profile_id)
                 and job['validation']['numerical']['status']=='not_evaluated'
                 and job['validation']['qualification']['sha256']==expected_qualification_sha256
                 and provenance['schema']=='aureon.cloud.provenance.v2'
                 and provenance['validation']==job['validation'], 'Guarded identity/validation mismatch')
        for name in ('class_map.tif','valid_mask.tif'):
            _require(provenance['files'][name]['sha256']==file_hash(output/name),'Provenance artifact hash differs')
        import rasterio
        import numpy as np
        with rasterio.open(output/'class_map.tif') as classes, rasterio.open(output/'valid_mask.tif') as mask:
            _require(classes.width==mask.width<=4096 and classes.height==mask.height<=4096
                     and classes.count==mask.count==1 and classes.dtypes==mask.dtypes==('uint8',)
                     and classes.nodata==255 and mask.nodata==0 and classes.crs==mask.crs
                     and str(classes.crs)==provenance['grid']['crs'] and classes.transform==mask.transform
                     and list(classes.transform)[:6]==provenance['grid']['transform']
                     and classes.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')=='COG'
                     and mask.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')=='COG', 'Downloaded COG grid/layout invalid')
            for row in range(0,classes.height,256):
                for col in range(0,classes.width,256):
                    window = rasterio.windows.Window(col,row,min(256,classes.width-col),min(256,classes.height-row))
                    values, validity = classes.read(1,window=window),mask.read(1,window=window)
                    _require(np.isin(validity,[0,1]).all() and (values[validity==1]<10).all()
                             and (values[validity==0]==255).all()
                             and np.array_equal(classes.read_masks(1,window=window)>0,validity==1), 'Downloaded COG mask/classes invalid')
        receipt = {'schema':'aureon.cloud.download.v2','job_id':identifier,'hashes_verified':True,
                   'cog_grid_masks_verified':True,'numerical_comparison':'not_evaluated',
                   'qualification_sha256':expected_qualification_sha256,'artifacts':listing['artifacts']}
        # Avoid retaining expiring capability strings in the local audit receipt.
        receipt['artifacts'] = {n:{k:v for k,v in d.items() if k!='url'} for n,d in listing['artifacts'].items()}
        with (output/'download.json').open('x') as stream:
            json.dump(receipt,stream,indent=2)
        return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument('--token-file',type=Path)
    credentials.add_argument('--credentials',type=Path,help='Private JSON emitted by local init')
    parser.add_argument('--tenant',help='Tenant name in --credentials')
    parser.add_argument('--operation',choices=('submit','status','health','cancel'),default='submit')
    parser.add_argument('--job-id',help='Existing logical job for queued-only cancellation')
    parser.add_argument('--spec',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--idempotency-key')
    args = parser.parse_args(argv)
    if args.operation=='submit' and not all((args.spec,args.output,args.idempotency_key)):
        parser.error('submit requires --spec --output --idempotency-key')
    if args.operation=='cancel' and not args.job_id:
        parser.error('cancel requires --job-id; running jobs are not interrupted')
    if args.credentials:
        if not args.tenant:
            parser.error('--credentials requires --tenant')
        client = CloudClient(args.url,token=_decode(args.credentials.read_bytes())[args.tenant])
    else:
        client = CloudClient.from_token_file(args.url,args.token_file)
    if args.operation!='submit':
        if args.operation=='status': result=client.status_snapshot()
        elif args.operation=='health': result=client.health()
        else: result=client.cancel_job(args.job_id)
        print(json.dumps(result,indent=2))
        return 1 if args.operation=='health' and not result['control_plane']['ready'] else 0
    profile = client.json('GET','/v1/models')['models'][0]['profiles'][0]
    _require(profile['execution']=='real','Use a qualified real state for COG delivery')
    identifier = client.upload_scene(args.spec)
    print(json.dumps({'input_id':identifier}),flush=True)
    prepared = client.wait_input(identifier)
    job = client.submit_input(identifier,profile['id'],args.idempotency_key)
    print(json.dumps({'job_id':job['id']}),flush=True)
    client.wait_job(job['id'])
    result = client.download_job(job['id'],args.output,expected_manifest_sha256=prepared['prepared_manifest_sha256'],
        expected_qualification_sha256=profile['qualification_sha256'])
    print(json.dumps({'job_id':job['id'],'download':str(args.output),'verified':result['hashes_verified']}))


if __name__=='__main__':
    raise SystemExit(main())
