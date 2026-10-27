"""Prepared-input async client; explicit reads/uploads and verified durable downloads."""
import argparse
import http.client
import json
import os
from pathlib import Path
import time
from .verified_files import VerifiedFiles
from ..cloud.client import CloudClient
from ..cloud.control import file_hash,CONTRACT
from ..client import _require,_decode
from .inputs import declarations,member

class Client(CloudClient):
    def upload_prepared(self,manifest,*,task,notify=None):
        start=time.monotonic();manifest=Path(manifest).resolve(strict=True);value=json.loads(manifest.read_bytes())
        records=declarations(value,task);paths={'manifest.json':manifest}
        for name,record in records.items():
            p=(manifest.parent/name).resolve(strict=True)
            _require(p.is_relative_to(manifest.parent) and p.is_file() and file_hash(p)==record['sha256'],'Prepared member identity invalid')
            paths[name]=p
        mapping={name:'part'+str(i)+p.suffix for i,(name,p) in enumerate(paths.items())}
        by_upload={mapping[name]:p for name,p in paths.items()}
        body={'spec':{'schema':'aureon.prepared_upload.v1','task':task,'manifest':'manifest.json','members':mapping},
          'files':[{'name':n,'bytes':p.stat().st_size,'sha256':file_hash(p)} for n,p in by_upload.items()]}
        created=self.json('POST','/v1/inputs',payload=body)
        if notify:notify({'event':'input_created','input_id':created['id'],'manifest_sha256':file_hash(manifest)})
        for item in created['uploads']:
            _require(item['name'] in by_upload and item['url'].startswith('/v1/inputs/'+created['id']+'/files/'+item['name']+'?ticket='),'Invalid upload target')
            cls=http.client.HTTPSConnection if self._scheme=='https' else http.client.HTTPConnection
            conn=cls(self._host,self._port,timeout=self.timeout)
            try:
                with by_upload[item['name']].open('rb') as stream:
                    conn.request('PUT',item['url'],body=stream,headers={'Authorization':'Bearer '+self._token,'Content-Type':'application/octet-stream',
                      'Content-Length':str(item['bytes']),'Connection':'close'})
                    response=conn.getresponse();reply=_decode(self._read(response,65536))
                    _require(response.status==200 and reply.get('sha256')==item['sha256'],'Upload failed')
            finally:conn.close()
        self.json('POST','/v1/inputs/'+created['id']+'/complete',payload={})
        return {'input_id':created['id'],'manifest_sha256':file_hash(manifest),'upload_seconds':time.monotonic()-start,'files':len(paths)}

    def submit_model(self,input_id,*,model,profile,task,key,seed=17):
        return self.json('POST','/v1/jobs',payload={'input_id':input_id,'model_id':model,'profile_id':profile,'task':task,
          'parameters':{'seed':seed} if task=='generation' else {}},key=key)

    def download_result(self,identifier,output,*,expected_manifest_sha256,model,task):
        start=time.monotonic();output=Path(output).absolute();_require(not output.exists() and output.parent.is_dir(),'Fresh download directory required')
        listing=self.json('GET','/v1/jobs/'+identifier+'/artifacts');_require(listing['job_id']==identifier,'Job identity mismatch')
        names=set(listing['artifacts']);fake=names=={'simulation.json'}
        expected={'job.json','native-job.json','provenance.json'}|({'class_map.tif','valid_mask.tif'} if task=='scene' else {'synthetic-s1grd.tif','raw.npy','tokens.npy'})
        _require(fake or names==expected,'Unexpected artifact contract')
        store=VerifiedFiles(output,max_total_bytes=CONTRACT['max_artifact_bytes']*len(names),max_files=len(names))
        saved={}
        for name,item in listing['artifacts'].items():
            member(name);_require('/' not in name and item['url'].startswith('/v1/jobs/'+identifier+'/artifacts/'+name+'?ticket='),'Invalid artifact path')
            with self._response('GET',item['url']) as response:data=self._read(response,CONTRACT['max_artifact_bytes'])
            try:saved[name]=store.save(name,data,expected=item)
            except ValueError as exc:
                _require(False,'Artifact hash or saved-file identity mismatch: '+str(exc))
            if name not in {'job.json','simulation.json'}:store.data.pop(name)
            data=None
        if fake:
            receipt=json.loads(store.data['simulation.json'])
            _require(receipt['model_executed'] is False and receipt['input_manifest_sha256']==expected_manifest_sha256,'Fake identity mismatch')
        else:
            receipt=json.loads(store.data['job.json'])
            _require(receipt['job_id']==identifier and receipt['model_id']==model and receipt['task']==task and receipt['input_manifest_sha256']==expected_manifest_sha256,'Result binding mismatch')
            _require(set(receipt['artifacts'])==names-{'job.json'},'Missing native bindings')
            for name,item in receipt['artifacts'].items():
                _require(saved[name]['sha256']==item['sha256'] and saved[name]['bytes']==item['bytes'],'Native export binding mismatch')
        proof={'schema':'aureon.async_download.v1','job_id':identifier,'hashes_verified':True,'fake':fake,'model_id':model,'task':task,
          'input_manifest_sha256':expected_manifest_sha256,'artifacts':saved,'download_seconds':time.monotonic()-start,
          'persisted_files_reopened':False,'persisted_bytes_read_back':True,
          'saved_byte_verification':'exclusive_unbuffered_descriptor_and_named_identity',
          'numerical_comparison':'not_evaluated_by_transport','COG_decoding':'consumer_audit_separate',
          'original_to_download':{} if fake else {r['original_path']:saved[n]['path'] for n,r in receipt['artifacts'].items()}}
        with (output/'download.json').open('x') as f:json.dump(proof,f,indent=2)
        store.release()
        return proof

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',required=True);p.add_argument('--credentials',type=Path,required=True);p.add_argument('--tenant',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ['models','health','ready','drain']:sub.add_parser(command)
    for command in ['status','cancel']:
        s=sub.add_parser(command);s.add_argument('--job-id',required=True)
    run=sub.add_parser('run')
    for key in ['manifest','model','profile','task','idempotency-key','output']:run.add_argument('--'+key,required=True)
    run.add_argument('--seed',type=int,default=17)
    run.add_argument('--input-id',help='Explicitly reuse an already uploaded input; never reupload on uncertain submission')
    run.add_argument('--recovery-file',type=Path,help='Fresh JSONL file for safe request/input/job IDs; no credentials')
    args=p.parse_args(argv)
    credentials=json.loads(args.credentials.read_bytes())
    token=credentials['key'] if credentials.get('tenant')==args.tenant and 'key' in credentials else credentials[args.tenant]
    client=Client(args.url,token=token)
    if args.command=='run':
        before=time.monotonic();recovery=args.recovery_file.open('x') if args.recovery_file else None
        if recovery:os.chmod(args.recovery_file,0o600)
        def progress(value):
            print(json.dumps(value),flush=True)
            if recovery:recovery.write(json.dumps(value)+'\n');recovery.flush();os.fsync(recovery.fileno())
        try:
            progress({'event':'request','model':args.model,'task':args.task,'profile':args.profile,'seed':args.seed,
              'idempotency_key':args.idempotency_key,'manifest_sha256':file_hash(Path(args.manifest))})
            if args.input_id:
                upload={'input_id':args.input_id,'manifest_sha256':file_hash(Path(args.manifest)),'reused_input':True,'upload_seconds':0.}
                progress({'event':'input_reused',**upload})
            else:upload=client.upload_prepared(args.manifest,task=args.task,notify=progress)
            prepared=client.wait_input(upload['input_id']);_require(prepared['prepared_manifest_sha256']==upload['manifest_sha256'],'Prepared manifest identity changed')
            progress({'event':'input_ready','input_id':upload['input_id']})
            job=client.submit_model(upload['input_id'],model=args.model,profile=args.profile,task=args.task,key=args.idempotency_key,seed=args.seed)
            progress({'event':'job_submitted','input_id':upload['input_id'],'job_id':job['id'],'reused':job['reused']})
            client.wait_job(job['id']);result=client.download_result(job['id'],args.output,expected_manifest_sha256=upload['manifest_sha256'],model=args.model,task=args.task)
            result.update(upload=upload,upload_through_download_seconds=time.monotonic()-before)
        finally:
            if recovery:recovery.close()
    elif args.command=='health':result=client.health()
    elif args.command=='cancel':result=client.cancel_job(args.job_id)
    elif args.command=='status':result=client.json('GET','/v1/jobs/'+args.job_id)
    elif args.command=='drain':result=client.json('POST','/v1/drain',payload={})
    else:result=client.json('GET','/v1/'+args.command)
    print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
