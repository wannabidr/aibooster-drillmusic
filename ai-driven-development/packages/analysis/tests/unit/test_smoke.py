"""Smoke tests for the analysis package."""


def test_package_imports():
    """Verify the analysis package can be imported."""
    import src

    assert src is not None
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_domain_layer_exists():
    """Verify domain layer modules exist."""
    import src.domain
    import src.domain.entities
    import src.domain.ports
    import src.domain.services
    import src.domain.value_objects

    assert src.domain is not None


def test_application_layer_exists():
    """Verify application layer modules exist."""
    import src.application
    import src.application.dto
    import src.application.services
    import src.application.use_cases

    assert src.application is not None


def test_infrastructure_layer_exists():
    """Verify infrastructure layer modules exist."""
    import src.infrastructure
    import src.infrastructure.analyzers
    import src.infrastructure.fingerprint
    import src.infrastructure.parsers
    import src.infrastructure.persistence

    assert src.infrastructure is not None


def test_interface_layer_exists():
    """Verify interface layer modules exist."""
    import src.interface

    assert src.interface is not None
