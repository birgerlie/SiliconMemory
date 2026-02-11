"""Custom exceptions for Silicon Memory."""


class MemoryError(Exception):
    """Base exception for memory operations."""

    pass


class BeliefConflictError(MemoryError):
    """Raised when beliefs conflict."""

    def __init__(self, belief_id: str, conflicting_ids: list[str], message: str = ""):
        self.belief_id = belief_id
        self.conflicting_ids = conflicting_ids
        super().__init__(
            message or f"Belief {belief_id} conflicts with: {conflicting_ids}"
        )


class StorageError(MemoryError):
    """Raised when storage operations fail."""

    pass


class ValidationError(MemoryError):
    """Raised when validation fails."""

    pass


class ExternalSourceError(MemoryError):
    """Raised when external source operations fail."""

    def __init__(self, source_id: str, message: str = ""):
        self.source_id = source_id
        super().__init__(message or f"External source error: {source_id}")


class LLMError(MemoryError):
    """Raised when LLM operations fail."""

    pass


class VerificationError(MemoryError):
    """Raised when verification fails."""

    def __init__(self, belief_id: str, reason: str):
        self.belief_id = belief_id
        self.reason = reason
        super().__init__(f"Verification failed for {belief_id}: {reason}")


class AuthorizationError(MemoryError):
    """Raised when authorization fails."""

    def __init__(
        self,
        user_id: str,
        resource_id: str,
        permission: str,
        reason: str = "",
    ):
        self.user_id = user_id
        self.resource_id = resource_id
        self.permission = permission
        self.reason = reason
        message = (
            f"User '{user_id}' not authorized for '{permission}' "
            f"on resource '{resource_id}'"
        )
        if reason:
            message += f": {reason}"
        super().__init__(message)


class SecurityError(MemoryError):
    """Raised when a security violation occurs."""

    def __init__(self, message: str, details: dict | None = None):
        self.details = details or {}
        super().__init__(message)


class TenantIsolationError(SecurityError):
    """Raised when tenant isolation is violated."""

    def __init__(self, user_tenant: str, resource_tenant: str):
        self.user_tenant = user_tenant
        self.resource_tenant = resource_tenant
        super().__init__(
            f"Tenant isolation violation: user from '{user_tenant}' "
            f"cannot access resource from '{resource_tenant}'"
        )


class ConsentRequiredError(SecurityError):
    """Raised when required consent is missing."""

    def __init__(self, consent_type: str, resource_id: str | None = None):
        self.consent_type = consent_type
        self.resource_id = resource_id
        message = f"Consent required: {consent_type}"
        if resource_id:
            message += f" for resource '{resource_id}'"
        super().__init__(message)
