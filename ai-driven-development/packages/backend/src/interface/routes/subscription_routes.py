"""FastAPI routes for Stripe subscription management.

Endpoints:
- POST /subscription/checkout - Create a checkout session
- POST /subscription/portal  - Create a customer portal session
- POST /subscription/webhook - Handle Stripe webhooks
- GET  /subscription/status  - Get current subscription status
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from src.domain.entities.user import User
from src.infrastructure.stripe.stripe_checkout_service import StripeCheckoutService
from src.infrastructure.stripe.stripe_webhook_handler import StripeWebhookHandler
from src.interface.middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subscription")


class CheckoutRequest(BaseModel):
    plan: str  # "monthly" or "annual"


class CheckoutResponse(BaseModel):
    checkout_url: str


class PortalResponse(BaseModel):
    portal_url: str


class SubscriptionStatusResponse(BaseModel):
    tier: str
    has_contributed: bool
    stripe_customer_id: str | None


def _get_checkout_service() -> StripeCheckoutService:
    """Override this dependency in app factory."""
    raise HTTPException(status_code=501, detail="Stripe not configured")


def _get_webhook_handler() -> StripeWebhookHandler:
    """Override this dependency in app factory."""
    raise HTTPException(status_code=501, detail="Stripe not configured")


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    body: CheckoutRequest,
    current_user: User = Depends(get_current_user),
    checkout_service: StripeCheckoutService = Depends(_get_checkout_service),
) -> CheckoutResponse:
    """Create a Stripe Checkout session for Pro subscription."""
    if body.plan not in ("monthly", "annual"):
        raise HTTPException(status_code=400, detail="Plan must be 'monthly' or 'annual'")

    url = await checkout_service.create_checkout_session(current_user, body.plan)
    return CheckoutResponse(checkout_url=url)


@router.post("/portal", response_model=PortalResponse)
async def create_portal(
    current_user: User = Depends(get_current_user),
    checkout_service: StripeCheckoutService = Depends(_get_checkout_service),
) -> PortalResponse:
    """Create a Stripe Customer Portal session for subscription management."""
    if not current_user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No active subscription to manage")

    url = await checkout_service.create_portal_session(current_user)
    return PortalResponse(portal_url=url)


@router.post("/webhook", status_code=200)
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(alias="Stripe-Signature"),
    handler: StripeWebhookHandler = Depends(_get_webhook_handler),
) -> dict:
    """Handle incoming Stripe webhook events."""
    payload = await request.body()

    try:
        event = handler.verify_and_parse(payload, stripe_signature)
    except Exception:
        logger.warning("Invalid Stripe webhook signature")
        raise HTTPException(status_code=400, detail="Invalid signature") from None

    await handler.handle_event(event)
    return {"status": "ok"}


@router.get("/status", response_model=SubscriptionStatusResponse)
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
) -> SubscriptionStatusResponse:
    """Get the current user's subscription status."""
    return SubscriptionStatusResponse(
        tier=current_user.subscription_tier.value,
        has_contributed=current_user.has_contributed,
        stripe_customer_id=current_user.stripe_customer_id,
    )
