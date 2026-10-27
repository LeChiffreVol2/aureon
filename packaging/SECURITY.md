# Private security reporting

Report suspected vulnerabilities privately to **norviccompanylimited@gmail.com**,
the owner-confirmed channel for AUREON Foundation by Norvique (Norvique Company).
It is also the Contact us address. No response-time, remediation or availability
SLA is promised. This text does not send any email or establish a hosted service.

Include exact package/wheel hash, selected profile, affected SDK/API route,
sanitized reproduction, impact and receipt identifiers. Do not put credentials,
imagery or exploit material in public issues. Arrange an appropriate private
transfer with the operator before sharing sensitive material.

This public-alpha release is for trusted local assets and operator-controlled
inputs. Checkpoints must match pinned identities; do not use arbitrary pickle,
remote Python code, untrusted factories or unrestricted upstream loading APIs.
The bundle converter verifies size/hash before CPU `weights_only=True` loading.
Keep HTTPS/authentication and access controls appropriate before external serving;
the local loopback listener is not a production ingress boundary.

Dependency locks establish reproducible identity, not absence of vulnerabilities.
The existing review recorded16 advisory matches in ten groups and retained the
qualified pins for restricted local inference. Excluded arbitrary-loading APIs
and residual uncertainty remain; no silent upgrade or vulnerability-free claim
is made. A materially broader hostile multi-tenant scope needs its own review.
This release does not change dependency pins or distribute their binaries.

Input validation includes declared bands/units/grid/masks, finite-size bounds,
hashes and provenance. Outputs preserve raw/numerical and transport-verification
distinctions. Back up service state/credentials privately, retain idempotency keys
on uncertain submissions, and never infer another tenant's result from a job ID.
See [support](SUPPORT.md), [local serving](ASYNC-SERVICE.md), and the supplied
license/NOTICE for third-party scope. Publication does not enlarge the documented local security/support boundary.
