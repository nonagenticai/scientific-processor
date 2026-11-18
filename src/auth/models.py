"""User models for license-token authentication with tenant + subject isolation."""

from pydantic import BaseModel, Field


class User(BaseModel):
    """User model with tenant + subject isolation.

    Two scoping axes:
    - tenant_id: the organization that owns the data
    - subject_id: the user (Subject UUID from an introspected license) within
      the tenant. Populated ONLY from an introspected license payload — never
      from a client-supplied X-Subject-Id header.

    Empty/None subject_id is the "tenant-only caller": subject-scoped reads
    collapse to the tenant. scientific-processor's RLS is tenant-only today
    (app.current_tenant GUC), so subject_id is carried for parity with the
    fleet auth contract but is not itself an RLS dimension here.
    """

    sub: str = Field(..., description="User ID")
    preferred_username: str = Field(default="anonymous", description="Username")
    email: str | None = None
    name: str | None = None
    realm_access: dict = Field(default_factory=dict)
    resource_access: dict = Field(default_factory=dict)
    tenant_id: str | None = Field(default=None, description="Tenant identifier")
    organization_id: str | None = Field(default=None, description="Organization ID")
    subject_id: str | None = Field(
        default=None,
        description="Subject UUID within the tenant (from an introspected license). None means tenant-only caller.",
    )
    tenant_name: str | None = Field(
        default=None,
        description="Human-readable tenant name. Falls back to effective_tenant_id.",
    )

    def model_post_init(self, __context) -> None:
        if self.tenant_id is None:
            self.tenant_id = self.organization_id or self.sub

    @property
    def effective_tenant_id(self) -> str:
        return self.tenant_id or self.organization_id or self.sub

    @property
    def effective_subject_id(self) -> str:
        """Subject (user) ID within the tenant.

        Falls back to ``effective_tenant_id`` when no subject was resolved from
        an introspected license, so tenant-only callers see their own
        tenant-scoped rows.
        """
        return self.subject_id or self.effective_tenant_id

    @property
    def effective_tenant_name(self) -> str:
        """Human-readable tenant name, falling back to ``effective_tenant_id``."""
        return self.tenant_name or self.effective_tenant_id

    def has_realm_role(self, role: str) -> bool:
        roles = self.realm_access.get("roles", [])
        return role in roles

    def has_client_role(self, client_id: str, role: str) -> bool:
        client_roles = self.resource_access.get(client_id, {}).get("roles", [])
        return role in client_roles

    def get_all_roles(self) -> list[str]:
        realm_roles: list[str] = self.realm_access.get("roles", [])
        client_roles: list[str] = []
        for client_data in self.resource_access.values():
            client_roles.extend(client_data.get("roles", []))
        return realm_roles + client_roles

    def is_admin(self) -> bool:
        return self.has_realm_role("admin") or self.has_realm_role("platform-admin")
