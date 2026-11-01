"""Loopback-only HTTP fixtures and socket doubles; no external network or models."""
from concurrent.futures import CancelledError
from dataclasses import replace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,socket,threading,time,unittest
from unittest.mock import patch
from aureon.sources.http_range import HTTPRangeSource,RangePolicy,RangeSourceError,SourceIdentity,_target,_addresses,_PinnedConnection

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args):pass
    def do_GET(self):
        self.server.requests.append((self.path,dict(self.headers)))
        mode=self.path.split('?')[0];start,end=map(int,self.headers['Range'][6:].split('-'))
        data=bytes(range(128));body=data[start:end+1]
        self.send_response(200 if mode=='/full' else 302 if mode.startswith('/redirect') else 206)
        if mode.startswith('/redirect'):
            self.send_header('Location',self.server.redirect);self.send_header('Content-Length','0');self.end_headers();return
        self.send_header('Content-Range',f'bytes {start+1 if mode=="/offset" else start}-{end}/{999999999999 if mode=="/large" else len(data)+1 if mode=="/different-size" else len(data)}')
        self.send_header('Content-Length',str(len(body)+1 if mode=='/length' else len(body)))
        if mode=='/duplicate':self.send_header('Content-Length',str(len(body)))
        if mode=='/encoded':self.send_header('Content-Encoding','gzip')
        if mode=='/chunked':self.send_header('Transfer-Encoding','chunked')
        if mode!='/missing':self.send_header('ETag','W/"weak"' if mode=='/weak' else self.server.etag)
        if mode=='/stall':self.send_header('Connection','close')
        self.end_headers();self.wfile.flush()
        if mode=='/stall':self.server.headers_sent.set();self.server.release.wait(4)
        if mode=='/truncated':body=body[:1];self.close_connection=True
        try:self.wfile.write(body);self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

class Ranges(unittest.TestCase):
    def setUp(self):
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.daemon_threads=True
        self.server.requests=[];self.server.etag='"v1"';self.server.headers_sent=threading.Event();self.server.release=threading.Event();self.server.redirect='/ok'
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start();self.sources=[]
        self.port=self.server.server_address[1]
        self.policy=RangePolicy(frozenset({'127.0.0.1','localhost'}),allowed_ports=frozenset({self.port}),allow_http=True,allowed_private_networks=('127.0.0.1/32',),max_range_bytes=16,max_total_bytes=32,chunk_bytes=4,timeout_seconds=3)
    def tearDown(self):
        for s in self.sources:s.close()
        self.server.release.set();self.server.shutdown();self.server.server_close();self.thread.join(3)
    def source(self,path='/ok',**kwargs):
        s=HTTPRangeSource(f'http://127.0.0.1:{self.port}{path}',policy=kwargs.pop('policy',self.policy),**kwargs);self.sources.append(s);return s
    def test_exact_ranges_pin_validator_and_send_conditional_request(self):
        s=self.source();self.assertEqual(s.read_range(4,8),bytes(range(4,12)))
        self.assertEqual(s.read_range(30,16),bytes(range(30,46)));self.assertEqual(s.identity,SourceIdentity('"v1"',128))
        self.assertEqual(self.server.requests[1][1]['If-Match'],'"v1"')
        self.assertEqual(s.provenance()['received_payload_bytes'],24)
    def test_invalid_response_never_falls_back_to_full_scene(self):
        for route in ['/full','/offset','/large','/length','/duplicate','/encoded','/chunked','/missing','/weak','/truncated']:
            with self.subTest(route=route):
                s=self.source(route)
                with self.assertRaises(RangeSourceError):s.read_range(1,8)
                count=len(self.server.requests)
                with self.assertRaises(RangeSourceError):s.read_range(1,8)
                self.assertEqual(count,len(self.server.requests));self.assertLessEqual(s.provenance()['received_payload_bytes'],8)
    def test_etag_change_invalidates_source(self):
        s=self.source();s.read_range(0,4);self.server.etag='"v2"'
        with self.assertRaisesRegex(RangeSourceError,'validator'):s.read_range(4,4)
        self.assertEqual(s.provenance()['received_payload_bytes'],4)
    def test_expected_identity_is_checked_before_reading_bytes(self):
        s=self.source(identity=SourceIdentity('"expected"',128))
        with self.assertRaises(RangeSourceError):s.read_range(0,4)
        self.assertEqual(s.provenance()['received_payload_bytes'],0)
    def test_changed_total_is_rejected_with_same_etag(self):
        s=self.source('/different-size',identity=SourceIdentity('"v1"',128))
        with self.assertRaisesRegex(RangeSourceError,'validator'):s.read_range(0,4)
        self.assertEqual(s.provenance()['received_payload_bytes'],0)
    def test_range_total_and_request_bounds(self):
        s=self.source();s.read_range(0,16);s.read_range(16,16)
        with self.assertRaisesRegex(RangeSourceError,'Total'):s.read_range(32,1)
        self.assertEqual(len(self.server.requests),2)
        for off,length in [(-1,1),(0,17),(True,2),(0,0)]:
            with self.assertRaises(RangeSourceError):self.source().read_range(off,length)
        s=self.source(policy=replace(self.policy,max_requests=1));s.read_range(0,4)
        with self.assertRaisesRegex(RangeSourceError,'Request'):s.read_range(4,4)
    def test_redirect_same_origin_and_no_policy_default(self):
        with self.assertRaises(RangeSourceError):self.source('/redirect').read_range(0,8)
        s=self.source('/redirect',policy=replace(self.policy,redirects='same-origin'))
        self.assertEqual(s.read_range(0,8),bytes(range(8)))
    def test_cross_origin_redirect_drops_all_custom_credentials(self):
        self.server.redirect=f'http://localhost:{self.port}/ok'
        with self.assertRaises(RangeSourceError):self.source('/redirect',policy=replace(self.policy,redirects='same-origin')).read_range(0,4)
        s=self.source('/redirect?signature=SECRET',policy=replace(self.policy,redirects='allowlisted'),headers={'Authorization':'Bearer SECRET','X-Api-Key':'SECRET'})
        # Resolve both origins to this IPv4 fixture, without depending on the
        # host's IPv6 localhost ordering or granting a wider private policy.
        with patch('socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',self.port))]):
            self.assertEqual(s.read_range(0,4),bytes(range(4)))
        self.assertEqual(self.server.requests[-2][1]['Authorization'],'Bearer SECRET')
        self.assertNotIn('Authorization',self.server.requests[-1][1]);self.assertNotIn('X-Api-Key',self.server.requests[-1][1])
        self.assertNotIn('SECRET',repr(s)+json.dumps(s.provenance())+repr(s.identity))
        self.assertNotIn('/redirect',json.dumps(s.provenance()))
    def test_redirect_cycle_is_bounded(self):
        self.server.redirect='/redirect'
        s=self.source('/redirect',policy=replace(self.policy,redirects='same-origin',max_redirects=1))
        with self.assertRaises(RangeSourceError):s.read_range(0,8)
        self.assertEqual(len(self.server.requests),2)
    def test_private_destination_rejected_before_socket_request(self):
        s=self.source(policy=replace(self.policy,allowed_private_networks=()))
        with self.assertRaisesRegex(RangeSourceError,'Private'):s.read_range(0,4)
        self.assertFalse(self.server.requests)
    def test_cancel_before_read_does_not_resolve_or_reserve(self):
        event=threading.Event();event.set();s=self.source()
        with patch('socket.getaddrinfo',side_effect=AssertionError('must not resolve')):
            with self.assertRaises(CancelledError):s.read_range(0,4,cancel_event=event)
        self.assertEqual(s.provenance()['reserved_payload_bytes'],0)
    def stalled(self,action):
        s=self.source('/stall');errors=[]
        event=threading.Event()
        def run():
            try:s.read_range(0,8,cancel_event=event)
            except BaseException as e:errors.append(e)
        thread=threading.Thread(target=run);thread.start();self.assertTrue(self.server.headers_sent.wait(2))
        time.sleep(.05) # ensure HTTPResponse has consumed Connection: close headers
        start=time.perf_counter()
        if action=='event':event.set()
        else:getattr(s,action)()
        thread.join(1)
        self.assertFalse(thread.is_alive(),'must interrupt response body even after connection.sock detaches')
        self.assertLess(time.perf_counter()-start,1);self.assertIsInstance(errors[0],CancelledError)
        self.assertIsNone(s._active)
    def test_cancel_interrupts_detached_connection_close_stalled_body(self):self.stalled('cancel')
    def test_close_interrupts_detached_connection_close_stalled_body(self):self.stalled('close')
    def test_external_event_interrupts_detached_stalled_body(self):self.stalled('event')
    def test_stalled_read_still_has_inactivity_timeout(self):
        s=self.source('/stall',policy=replace(self.policy,timeout_seconds=.2))
        start=time.perf_counter()
        with self.assertRaisesRegex(RangeSourceError,'transport'):s.read_range(0,4)
        self.assertLess(time.perf_counter()-start,1)
        self.assertIsNone(s._active)
    def test_concurrent_read_is_rejected(self):
        s=self.source('/stall');errors=[]
        def run():
            try:s.read_range(0,8)
            except BaseException as e:errors.append(e)
        thread=threading.Thread(target=run);thread.start();self.assertTrue(self.server.headers_sent.wait(2))
        with self.assertRaisesRegex(RangeSourceError,'active'):s.read_range(0,4)
        s.cancel();thread.join(2);self.assertFalse(thread.is_alive())

class Policy(unittest.TestCase):
    def test_policy_rejects_unsafe_defaults_bounds_and_mutation(self):
        p=RangePolicy(frozenset({'example.com'}))
        for url in ['http://example.com/a','https://user:SECRET@example.com/a','https://example.com/a#SECRET','https://other.com/a','https://example.com:444/a','https://example.com/\r\nX:SECRET']:
            with self.assertRaises(RangeSourceError) as e:HTTPRangeSource(url,policy=p)
            self.assertNotIn('SECRET',str(e.exception))
        for kwargs in [dict(max_total_bytes=0),dict(max_requests=True),dict(timeout_seconds=float('nan')),dict(allowed_hosts=frozenset({'*.example.com'}))]:
            with self.assertRaises(RangeSourceError):replace(p,**kwargs)
        for headers in [{'Range':'bytes=0-9'},{'Authorization':'x\r\nSECRET'},{'Cookie':'SECRET'},{'X-Key':'x','x-key':'y'}]:
            with self.assertRaises(RangeSourceError):HTTPRangeSource('https://example.com/a',policy=p,headers=headers)
    def test_dns_mixed_private_and_mapped_multicast_unspecified_rejected(self):
        p=RangePolicy(frozenset({'example.com'}),allowed_private_networks=('0.0.0.0/0',))
        target=_target('https://example.com/x',p)
        for addr in ['::ffff:224.0.0.1','::ffff:0.0.0.0','ff02::1','::']:
            with patch('socket.getaddrinfo',return_value=[(socket.AF_INET6,socket.SOCK_STREAM,6,'',(addr,443,0,0))]):
                with self.assertRaises(RangeSourceError):_addresses(target,p)
        with patch('socket.getaddrinfo',return_value=[(socket.AF_INET,socket.SOCK_STREAM,6,'',('8.8.8.8',443)),(socket.AF_INET,socket.SOCK_STREAM,6,'',('127.0.0.1',443))]):
            with self.assertRaises(RangeSourceError):_addresses(target,replace(p,allowed_private_networks=()))
    def test_connection_pins_ip_and_uses_original_tls_hostname_without_second_dns(self):
        p=RangePolicy(frozenset({'example.com'}));target=_target('https://example.com/x',p)
        with patch('socket.socket') as factory,patch('ssl.create_default_context') as context,patch('socket.getaddrinfo',side_effect=AssertionError('second DNS')):
            raw=factory.return_value;raw.getpeername.return_value=('8.8.8.8',443)
            conn=_PinnedConnection(target,(socket.AF_INET,('8.8.8.8',443)),1.);conn.connect()
            raw.connect.assert_called_once_with(('8.8.8.8',443));context.return_value.wrap_socket.assert_called_once_with(raw,server_hostname='example.com')
            conn.finish();context.return_value.wrap_socket.return_value.close.assert_called()
if __name__=='__main__':unittest.main()
