"""WS-4 on-behalf-of vouching — the only thing that makes a bare tenant delegatable.

WHY THIS EXISTS
---------------
``AuthMiddleware``'s trusted-network branch accepts a BARE, uncredentialed
``X-Tenant-Id: <tenant>`` from any peer inside ``TRUSTED_CIDRS`` — which in
production is the whole pod CIDR (``10.0.0.0/8``). That was harmless while the
tenant only selected a DATA SCOPE inside scientific-processor: RLS scopes the
rows either way and every downstream re-derives its own authorisation.

It stopped being harmless the moment this service became a DELEGATOR. Since the
modality-gateway flip, ``clients.pytorch_backend._embedding_headers`` stamps
``X-Effective-Tenant-Id: <that same bare header>`` on the outbound
``/api/ml/embeddings`` and ``/api/ml/gnn/inference`` calls. The SA subject
``019f83f2-ddd1-779f-9c70-7a0c6de1dacb`` (svc-scientific-processor) is on the
gateway's ``AUTH_DELEGATING_SUBJECTS``, so the gateway BELIEVES that header: it
resolves that tenant's BYO LLM provider credentials and bills that tenant. Net
effect before this module existed — any unauthenticated pod in the cluster could
name a victim tenant and spend the victim's LLM credentials.

THE REMEDY (ported from hirag_service / kag / kag_graph_service)
----------------------------------------------------------------
A caller that wants scientific-processor to ACT FOR a tenant it names must prove
it is entitled to: it presents its OWN service license token in
``Authorization: Bearer <token>``, we introspect it via
``kag_common.auth.introspect_license``, and the resulting ``subject_id`` must
appear in ``SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS``. Only then is the bare
``X-Tenant-Id`` promoted from ``unvouched-header`` to ``vouched-onbehalf`` — the
one non-license-token provenance the ML client will delegate.

SCOPE NOTE — the ``Authorization`` header
-----------------------------------------
This service deliberately does NOT treat ``Authorization`` as its own
authentication: license tokens travel only in ``X-Tenant-Id: Bearer <token>``
(see ``auth/middleware.py``). This module is the ONLY place ``Authorization`` is
read, and reading it can only ever WIDEN what a request may do — never reject
it. A caller that sends an unrelated ``Authorization`` (e.g. a forwarded
Keycloak JWT) simply fails to vouch and is served exactly as before.

FAIL-CLOSED CHECKLIST (every one of these ⇒ NOT vouched)
--------------------------------------------------------
- ``SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS`` empty or unset (the DEFAULT —
  vouching is OFF until gitops opts in, and OFF means nothing is ever delegated)
- ``LICENSE_INTROSPECT_URL`` unset
- no ``Authorization`` header, or not ``Bearer``-shaped, or an empty token
- the license introspects as inactive / malformed / unknown
- introspection times out or the licenses endpoint is down
- the introspected subject is empty, or is not on the allowlist

``kag_common.auth.introspect_license`` is reused verbatim rather than
hand-rolled (this service already depends on ``kag-common>=0.19.1``), which also
inherits its cache discipline: ``active`` payloads are memoised for 5 min,
``active: false`` briefly, and a TRANSPORT fault returns ``None`` WITHOUT being
memoised — so a blip in the licenses service cannot pin a legitimate delegator
into a denial for the length of a cache TTL.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from kag_common.auth import introspect_license

if TYPE_CHECKING:
    from starlette.requests import Request

logger: logging.Logger = logging.getLogger(__name__)

_BEARER_PREFIX = "bearer "

#: CSV env var naming the service ``subject_id``s permitted to vouch for a bare
#: tenant. Uses this repo's ``SCIENTIFIC_PROCESSOR_`` config prefix (see
#: ``src/config.py``); set on the scientific-processor Deployment in
#: gitops-production.
ONBEHALF_SUBJECTS_ENV = "SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS"


def onbehalf_subjects() -> frozenset[str]:
    """The allowlist of service ``subject_id``s permitted to vouch.

    Read at CALL time (not import time) so a config reload or a test override
    takes effect without re-importing the module — matching
    ``kag_common.auth.delegating_subjects``. Empty ⇒ vouching is off and nothing
    is ever delegated: the fail-safe default.
    """
    raw = os.getenv(ONBEHALF_SUBJECTS_ENV, "")
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


def _extract_bearer(request: Request) -> str | None:
    """The raw token from ``Authorization: Bearer <token>``, if shaped like one."""
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith(_BEARER_PREFIX):
        return None
    token = header[len(_BEARER_PREFIX) :].strip()
    return token or None


async def authorized_onbehalf(request: Request) -> bool:
    """True iff this request may DELEGATE the tenant it names.

    Requires ALL of: a non-empty ``SCIENTIFIC_PROCESSOR_ONBEHALF_SUBJECTS``
    allowlist, a configured ``LICENSE_INTROSPECT_URL``, an ``Authorization:
    Bearer`` header, an *active* license behind it, and an introspected
    ``subject_id`` on the allowlist. Any missing link ⇒ False.

    Never raises: a vouch that cannot be PROVEN is simply not a vouch, and the
    request continues on its original (unvouched) footing. This function can
    only WIDEN.
    """
    subjects = onbehalf_subjects()
    if not subjects:
        return False
    if not os.getenv("LICENSE_INTROSPECT_URL", ""):
        logger.warning(
            "WS4_VOUCH disabled: %s is set but LICENSE_INTROSPECT_URL is not — "
            "no request can be vouched, so nothing will be delegated",
            ONBEHALF_SUBJECTS_ENV,
        )
        return False

    token = _extract_bearer(request)
    if token is None:
        return False

    # introspect_license swallows transport faults and returns None (and does
    # NOT memoise that None) — so a licenses-service blip degrades to "not
    # vouched" for this request only, never a sticky denial.
    data = await introspect_license(token)
    if not data or not data.get("active"):
        return False

    raw_subject = data.get("subject_id")
    subject = str(raw_subject).strip() if raw_subject is not None else ""
    if not subject:
        return False
    if subject not in subjects:
        logger.warning("WS4_VOUCH refused: subject=%s is not on %s", subject, ONBEHALF_SUBJECTS_ENV)
        return False
    return True
