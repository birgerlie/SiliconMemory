"""Security configuration for Silicon Memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecurityConfig:
    """Security configuration settings.

    Controls default behaviors for privacy, access control,
    audit logging, and data retention.

    Example:
        >>> config = SecurityConfig(
        ...     enforce_tenant_isolation=True,
        ...     default_retention_days=365,
        ...     audit_read_operations=True,
        ... )
    """

    # Tenant isolation
    enforce_tenant_isolation: bool = True
    allow_cross_tenant_public: bool = True

    # Default privacy settings
    default_privacy_level: str = "private"  # private, workspace, public
    default_classification: str = "internal"

    # Retention settings
    default_retention_days: int | None = None
    max_retention_days: int | None = None
    enforce_retention_limits: bool = False

    # Consent requirements
    require_storage_consent: bool = False
    require_processing_consent: bool = False
    require_sharing_consent: bool = True

    # Audit settings
    audit_enabled: bool = True
    audit_read_operations: bool = False  # Can be verbose
    audit_write_operations: bool = True
    audit_delete_operations: bool = True
    audit_retention_days: int = 90

    # Access control
    enable_abac: bool = True
    admin_bypass_enabled: bool = True

    # Forgetting
    allow_selective_forget: bool = True
    allow_session_forget: bool = True
    allow_gdpr_forget: bool = True
    hard_delete_on_forget: bool = True

    # Transparency
    enable_provenance_tracking: bool = True
    enable_access_logging: bool = True

    # Data export/import
    allow_data_export: bool = True
    allow_data_import: bool = True
    require_export_consent: bool = False

    # PII detection (patterns to check for "do not remember")
    pii_patterns: list[str] = field(default_factory=lambda: [
        r"\d{3}-\d{2}-\d{4}",  # SSN
        r"\d{16}",  # Credit card (basic)
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ])

    # Additional custom settings
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a custom setting."""
        return self.custom_settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Set a custom setting."""
        self.custom_settings[key] = value

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.default_retention_days and self.max_retention_days:
            if self.default_retention_days > self.max_retention_days:
                issues.append(
                    "default_retention_days cannot exceed max_retention_days"
                )

        if self.audit_retention_days < 1:
            issues.append("audit_retention_days must be at least 1")

        valid_privacy = {"private", "workspace", "public"}
        if self.default_privacy_level not in valid_privacy:
            issues.append(f"default_privacy_level must be one of {valid_privacy}")

        valid_classification = {"public", "internal", "confidential", "personal", "sensitive"}
        if self.default_classification not in valid_classification:
            issues.append(f"default_classification must be one of {valid_classification}")

        return issues

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enforce_tenant_isolation": self.enforce_tenant_isolation,
            "allow_cross_tenant_public": self.allow_cross_tenant_public,
            "default_privacy_level": self.default_privacy_level,
            "default_classification": self.default_classification,
            "default_retention_days": self.default_retention_days,
            "max_retention_days": self.max_retention_days,
            "enforce_retention_limits": self.enforce_retention_limits,
            "require_storage_consent": self.require_storage_consent,
            "require_processing_consent": self.require_processing_consent,
            "require_sharing_consent": self.require_sharing_consent,
            "audit_enabled": self.audit_enabled,
            "audit_read_operations": self.audit_read_operations,
            "audit_write_operations": self.audit_write_operations,
            "audit_delete_operations": self.audit_delete_operations,
            "audit_retention_days": self.audit_retention_days,
            "enable_abac": self.enable_abac,
            "admin_bypass_enabled": self.admin_bypass_enabled,
            "allow_selective_forget": self.allow_selective_forget,
            "allow_session_forget": self.allow_session_forget,
            "allow_gdpr_forget": self.allow_gdpr_forget,
            "hard_delete_on_forget": self.hard_delete_on_forget,
            "enable_provenance_tracking": self.enable_provenance_tracking,
            "enable_access_logging": self.enable_access_logging,
            "allow_data_export": self.allow_data_export,
            "allow_data_import": self.allow_data_import,
            "require_export_consent": self.require_export_consent,
            "pii_patterns": self.pii_patterns,
            "custom_settings": self.custom_settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityConfig":
        """Create from dictionary."""
        return cls(
            enforce_tenant_isolation=data.get("enforce_tenant_isolation", True),
            allow_cross_tenant_public=data.get("allow_cross_tenant_public", True),
            default_privacy_level=data.get("default_privacy_level", "private"),
            default_classification=data.get("default_classification", "internal"),
            default_retention_days=data.get("default_retention_days"),
            max_retention_days=data.get("max_retention_days"),
            enforce_retention_limits=data.get("enforce_retention_limits", False),
            require_storage_consent=data.get("require_storage_consent", False),
            require_processing_consent=data.get("require_processing_consent", False),
            require_sharing_consent=data.get("require_sharing_consent", True),
            audit_enabled=data.get("audit_enabled", True),
            audit_read_operations=data.get("audit_read_operations", False),
            audit_write_operations=data.get("audit_write_operations", True),
            audit_delete_operations=data.get("audit_delete_operations", True),
            audit_retention_days=data.get("audit_retention_days", 90),
            enable_abac=data.get("enable_abac", True),
            admin_bypass_enabled=data.get("admin_bypass_enabled", True),
            allow_selective_forget=data.get("allow_selective_forget", True),
            allow_session_forget=data.get("allow_session_forget", True),
            allow_gdpr_forget=data.get("allow_gdpr_forget", True),
            hard_delete_on_forget=data.get("hard_delete_on_forget", True),
            enable_provenance_tracking=data.get("enable_provenance_tracking", True),
            enable_access_logging=data.get("enable_access_logging", True),
            allow_data_export=data.get("allow_data_export", True),
            allow_data_import=data.get("allow_data_import", True),
            require_export_consent=data.get("require_export_consent", False),
            pii_patterns=data.get("pii_patterns", []),
            custom_settings=data.get("custom_settings", {}),
        )

    @classmethod
    def strict(cls) -> "SecurityConfig":
        """Create a strict security configuration.

        Suitable for environments with high security requirements.
        """
        return cls(
            enforce_tenant_isolation=True,
            allow_cross_tenant_public=False,
            default_privacy_level="private",
            default_classification="confidential",
            require_storage_consent=True,
            require_processing_consent=True,
            require_sharing_consent=True,
            audit_enabled=True,
            audit_read_operations=True,
            audit_write_operations=True,
            audit_delete_operations=True,
            admin_bypass_enabled=False,
            hard_delete_on_forget=True,
            enable_provenance_tracking=True,
            enable_access_logging=True,
            require_export_consent=True,
        )

    @classmethod
    def permissive(cls) -> "SecurityConfig":
        """Create a permissive security configuration.

        Suitable for development or low-security environments.
        """
        return cls(
            enforce_tenant_isolation=True,
            allow_cross_tenant_public=True,
            default_privacy_level="workspace",
            default_classification="internal",
            require_storage_consent=False,
            require_processing_consent=False,
            require_sharing_consent=False,
            audit_enabled=True,
            audit_read_operations=False,
            audit_write_operations=True,
            audit_delete_operations=True,
            admin_bypass_enabled=True,
            hard_delete_on_forget=True,
            require_export_consent=False,
        )
