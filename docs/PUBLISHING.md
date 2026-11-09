# Package-index publication procedure

GitHub currently distributes Best v0.1.0 (`e09e710c`). PyPI is not open.
The short command `pip install aureon-inference` becomes valid only after the
owner establishes and publishes the actual package there. Do not advertise it
as a working command before that event.

## Check the reviewed artifact

```sh
python3.12 tools/check_index_metadata.py --wheel /absolute/aureon_inference-0.1.0-py3-none-any.whl
```

The frozen Best wheel's embedded README still selects the historical `0fa5ef2b`
runtime and contains superseded delivery wording. This check therefore returns
exit code 2 for that wheel. Its executable payloads and existing FP32 acceptance
are unaffected. The current source README has been corrected, but editing source
does not change an already-built archive.

Before a PyPI upload, review a metadata-corrected artifact and verify its runtime
payloads against the selected Best. Updating metadata changes the wheel hash:
preserve the old archive/receipt and record the new identity explicitly. Do not
silently replace the frozen download, rewrite the original release tag, or reuse
historical benchmark percentages as a new measurement. Version/release selection
is an explicit maintainer decision. No replacement archive is produced by this
documentation update.

## Owner setup and publication

1. Create or use an owner-controlled PyPI account with 2FA, and confirm the
   `aureon-inference` name is available. This review does not reserve that name.
2. Configure a pending Trusted Publisher for the public GitHub repository and
   its reviewed publication workflow/environment. Configure TestPyPI separately
   for a rehearsal. Keep tokens and account recovery codes out of issues and Git.
3. Have CI produce or retrieve the reviewed artifact, verify wheel/source hashes,
   install it in a clean supported environment and run installed examples.
   Separate build/check jobs from the upload job. The publication job consumes
   those checked bytes, with explicit owner approval and short-lived OIDC access.
4. Confirm the index serves those exact bytes. Test the short installation command
   from outside a checkout. Then update release notes, current distribution state
   and public instructions together. Existing evidence retains its artifact scope.

The owner account/publisher and metadata review remain outstanding. No upload
workflow is enabled by default, and hosted API/payment activation is separate.

References: [PyPA publication guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
and [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
