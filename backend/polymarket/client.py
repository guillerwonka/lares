"""
Polymarket API client — wraps py-clob-client for 5-min BTC UP/DOWN markets.
Sync SDK calls wrapped with asyncio.to_thread.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams, MarketOrderArgs, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon


@dataclass
class MarketInfo:
    """Currently active 5-min BTC UP/DOWN market."""
    slug: str
    question: str
    condition_id: str
    token_up: str       # token_id for YES (UP)
    token_down: str     # token_id for NO (DOWN)
    end_time: int       # unix seconds when window ends
    window_start: int   # unix seconds when window started


class PolyClient:
    """Async wrapper around Polymarket CLOB + Gamma APIs."""

    def __init__(self) -> None:
        self._clob: Optional[ClobClient] = None
        self._http = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self._current_market: Optional[MarketInfo] = None
        self._last_market_ts: int = 0  # tracks which 5-min window
        self._relayer_api_key: str = ""

    async def connect(
        self,
        private_key: str,
        funder_address: str,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        relayer_api_key: str = "",
    ) -> None:
        """Initialize CLOB client with trading credentials."""
        from py_clob_client.clob_types import ApiCreds

        has_private_key = bool(private_key)
        has_api_creds = bool(api_key and api_secret and api_passphrase)

        def _init():
            if has_private_key:
                # Full L2: can sign orders + authenticate
                # signature_type=1 = POLY_PROXY (Polymarket proxy wallet)
                client = ClobClient(
                    host=CLOB_API,
                    key=private_key,
                    chain_id=CHAIN_ID,
                    signature_type=1,
                    funder=funder_address,
                )
                # Always derive fresh creds from private key for consistency
                creds = client.create_or_derive_api_creds()
                client.set_api_creds(creds)
            else:
                # L0 read-only: market discovery + prices only
                client = ClobClient(host=CLOB_API, chain_id=CHAIN_ID)
                if has_api_creds:
                    client.set_api_creds(ApiCreds(api_key, api_secret, api_passphrase))
            return client

        self._clob = await asyncio.to_thread(_init)
        self._relayer_api_key = relayer_api_key
        mode = "L2 (trading)" if has_private_key else "L0 (read-only)"
        logger.info("PolyClient connected %s (funder=%s)", mode, funder_address[:10] + "..." if funder_address else "none")

        # Ensure USDC allowance is set for trading
        if has_private_key:
            await self.ensure_allowance()

    def _ensure_http(self):
        """Recreate HTTP client if it was closed."""
        if self._http.is_closed:
            logger.warning("HTTP client was closed, recreating...")
            self._http = httpx.AsyncClient(
                timeout=30,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )

    async def disconnect(self) -> None:
        await self._http.aclose()
        logger.info("PolyClient disconnected")

    async def ensure_allowance(self, token_id: str = "") -> None:
        """Update balance allowance. If token_id given, also approve CONDITIONAL for selling."""
        if not self._clob:
            return
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = await asyncio.to_thread(self._clob.update_balance_allowance, params)
            logger.info("Allowance updated (COLLATERAL): %s", result)
        except Exception as e:
            logger.warning("update_balance_allowance(COLLATERAL) failed: %s", e)
        if token_id:
            try:
                params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
                result = await asyncio.to_thread(self._clob.update_balance_allowance, params)
                logger.info("Allowance updated (CONDITIONAL %s...): %s", token_id[:12], result)
            except Exception as e:
                logger.warning("update_balance_allowance(CONDITIONAL) failed: %s", e)

    async def get_balance(self) -> Optional[dict]:
        """Get current USDC balance and allowance status."""
        if not self._clob:
            return None
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            return await asyncio.to_thread(self._clob.get_balance_allowance, params)
        except Exception as e:
            logger.error("get_balance failed: %s", e)
            return None

    async def get_conditional_balance(self, token_id: str) -> Optional[str]:
        """Get CONDITIONAL token balance for a specific token."""
        if not self._clob:
            return None
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
            result = await asyncio.to_thread(self._clob.get_balance_allowance, params)
            return result.get("balance", "0") if result else "0"
        except Exception as e:
            logger.error("get_conditional_balance failed: %s", e)
            return None

    # ── Market discovery ──

    def _current_window_ts(self) -> int:
        """Get the start timestamp of the current 5-min window."""
        return int(time.time() // 300) * 300

    @staticmethod
    def _parse_token_ids(raw) -> list[str]:
        """Parse clobTokenIds — Gamma API returns them as a JSON string."""
        if isinstance(raw, str):
            return json.loads(raw)
        return raw if isinstance(raw, list) else []

    async def discover_current_market(self) -> Optional[MarketInfo]:
        """Find the active 5-min BTC UP/DOWN market via Gamma API."""
        window_ts = self._current_window_ts()

        # Return cached if same window
        if self._current_market and self._last_market_ts == window_ts:
            return self._current_market

        slug = f"btc-updown-5m-{window_ts}"
        try:
            resp = await self._http.get(
                f"{GAMMA_API}/events",
                params={"slug": slug, "limit": 1},
            )
            resp.raise_for_status()
            events = resp.json()

            if not events:
                # Try search by slug in markets endpoint
                resp2 = await self._http.get(
                    f"{GAMMA_API}/markets",
                    params={"slug": slug, "limit": 1},
                )
                resp2.raise_for_status()
                markets_data = resp2.json()
                if not markets_data:
                    logger.debug("No market found for slug %s", slug)
                    return None
                market = markets_data[0] if isinstance(markets_data, list) else markets_data
                tokens = self._parse_token_ids(market.get("clobTokenIds", []))
                if len(tokens) < 2:
                    return None
                self._current_market = MarketInfo(
                    slug=slug,
                    question=market.get("question", slug),
                    condition_id=market.get("conditionId", ""),
                    token_up=tokens[0],
                    token_down=tokens[1],
                    end_time=window_ts + 300,
                    window_start=window_ts,
                )
            else:
                event = events[0]
                markets = event.get("markets", [])
                if not markets:
                    return None
                # First market in the event
                m = markets[0]
                tokens = self._parse_token_ids(m.get("clobTokenIds", []))
                if len(tokens) < 2:
                    return None
                self._current_market = MarketInfo(
                    slug=slug,
                    question=event.get("title", slug),
                    condition_id=m.get("conditionId", ""),
                    token_up=tokens[0],
                    token_down=tokens[1],
                    end_time=window_ts + 300,
                    window_start=window_ts,
                )

            self._last_market_ts = window_ts
            logger.info(
                "Discovered market: %s (UP=%s..., DOWN=%s...)",
                self._current_market.question,
                self._current_market.token_up[:12],
                self._current_market.token_down[:12],
            )
            return self._current_market

        except Exception as e:
            logger.error("Market discovery failed: %s", e)
            return None

    # ── Pricing ──

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a token."""
        if not self._clob:
            return None
        try:
            result = await asyncio.to_thread(self._clob.get_midpoint, token_id)
            return float(result.get("mid", 0))
        except Exception as e:
            logger.error("get_midpoint failed: %s", e)
            return None

    async def get_spread(self, token_id: str) -> Optional[dict]:
        """Get bid/ask spread."""
        if not self._clob:
            return None
        try:
            return await asyncio.to_thread(self._clob.get_spread, token_id)
        except Exception as e:
            logger.error("get_spread failed: %s", e)
            return None

    async def get_orderbook(self, token_id: str) -> Optional[dict]:
        """Get full orderbook for depth analysis."""
        if not self._clob:
            return None
        try:
            return await asyncio.to_thread(self._clob.get_order_book, token_id)
        except Exception as e:
            logger.error("get_orderbook failed: %s", e)
            return None

    @staticmethod
    def _top_depth(entries: list, n: int = 3) -> float:
        """Sum USD depth at top N price levels."""
        total = 0.0
        for entry in entries[:n]:
            price = float(entry.price)
            size = float(entry.size)
            total += price * size
        return total

    async def check_liquidity(self, token_id: str, min_depth: float = 50.0) -> bool:
        """Check if there's enough depth to trade."""
        book = await self.get_orderbook(token_id)
        if not book:
            return False
        bid_depth = self._top_depth(book.bids)
        ask_depth = self._top_depth(book.asks)
        return min(bid_depth, ask_depth) >= min_depth

    # ── Trading ──

    async def place_limit_buy(
        self, token_id: str, price: float, size: float,
    ) -> Optional[dict]:
        """Place a GTC limit buy order (maker, 0% fee)."""
        if not self._clob:
            return None
        try:
            order_args = OrderArgs(
                price=round(price, 2),
                size=round(size, 1),
                side=BUY,
                token_id=token_id,
            )
            def _create_and_post():
                signed = self._clob.create_order(order_args)
                return self._clob.post_order(signed, OrderType.GTC)

            result = await asyncio.to_thread(_create_and_post)
            logger.info(
                "Limit buy posted: token=%s... price=%.2f size=%.1f → %s",
                token_id[:12], price, size, result,
            )
            return result
        except Exception as e:
            logger.error("place_limit_buy failed: %s", e)
            return None

    async def place_limit_sell(
        self, token_id: str, price: float, size: float,
    ) -> Optional[dict]:
        """Place a GTC limit sell order (for stop-loss exits)."""
        if not self._clob:
            return None
        try:
            # CRITICAL: truncate size DOWN to avoid exceeding on-chain balance
            safe_size = math.floor(size * 100) / 100  # truncate to 2 decimals
            order_args = OrderArgs(
                price=round(price, 2),
                size=safe_size,
                side=SELL,
                token_id=token_id,
            )
            def _create_and_post():
                signed = self._clob.create_order(order_args)
                return self._clob.post_order(signed, OrderType.GTC)

            result = await asyncio.to_thread(_create_and_post)
            logger.info(
                "Limit sell posted: token=%s... price=%.2f size=%.1f → %s",
                token_id[:12], price, size, result,
            )
            return result
        except Exception as e:
            logger.error("place_limit_sell failed: %s", e)
            return None

    async def place_market_sell(
        self, token_id: str, size: float, price: float = 0,
    ) -> Optional[dict]:
        """Place a FOK market sell order (taker, for stop-loss exits).
        Uses create_market_order which fills immediately against existing bids."""
        if not self._clob:
            return None
        try:
            # CRITICAL: truncate size DOWN to avoid exceeding on-chain balance
            safe_size = math.floor(size * 100) / 100  # truncate to 2 decimals
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=safe_size,
                side=SELL,
                price=round(price, 2) if price > 0 else 0,
                order_type=OrderType.FOK,
            )
            def _create_and_post():
                signed = self._clob.create_market_order(order_args)
                return self._clob.post_order(signed, OrderType.FOK)

            result = await asyncio.to_thread(_create_and_post)
            logger.info(
                "Market sell posted: token=%s... size=%.1f price=%.2f → %s",
                token_id[:12], size, price, result,
            )
            return result
        except Exception as e:
            logger.error("place_market_sell failed: %s", e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        if not self._clob:
            return False
        try:
            await asyncio.to_thread(self._clob.cancel, order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as e:
            logger.error("cancel_order failed: %s", e)
            return False

    async def cancel_all(self) -> bool:
        if not self._clob:
            return False
        try:
            await asyncio.to_thread(self._clob.cancel_all)
            logger.info("Cancelled all open orders")
            return True
        except Exception as e:
            logger.error("cancel_all failed: %s", e)
            return False

    async def get_open_orders(self) -> list:
        if not self._clob:
            return []
        try:
            result = await asyncio.to_thread(self._clob.get_orders)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error("get_open_orders failed: %s", e)
            return []

    async def get_order_details(self, order_id: str) -> Optional[dict]:
        """Fetch order details including fill price and matched size."""
        if not self._clob:
            return None
        try:
            return await asyncio.to_thread(self._clob.get_order, order_id)
        except Exception as e:
            logger.debug("get_order_details(%s) failed: %s", order_id[:12], e)
            return None

    async def get_market_resolution(self, slug: str) -> Optional[str]:
        """Query Gamma API to check if a market has resolved. Returns 'UP', 'DOWN', or None."""
        try:
            resp = await self._http.get(
                f"{GAMMA_API}/events",
                params={"slug": slug, "limit": 1},
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                return None
            markets = events[0].get("markets", [])
            if not markets:
                return None
            m = markets[0]
            # Check various resolution indicators
            outcome = m.get("outcome") or m.get("resolution")
            if not outcome:
                # Check if any token is marked as winner
                tokens = m.get("tokens", [])
                for i, t in enumerate(tokens):
                    if t.get("winner"):
                        return "UP" if i == 0 else "DOWN"
                return None
            # Polymarket 15m BTC UP/DOWN: token[0]=UP=Yes, token[1]=DOWN=No
            if outcome in ("Yes", "UP", "Up"):
                return "UP"
            elif outcome in ("No", "DOWN", "Down"):
                return "DOWN"
            return None
        except Exception as e:
            logger.debug("get_market_resolution(%s) failed: %s", slug, e)
            return None

    # ── Redemption ──

    async def redeem_positions(self, condition_id: str) -> Optional[str]:
        """Redeem winning conditional tokens for USDC via Polymarket relayer (gasless).

        Uses the relayer SDK to build + sign a Safe transaction, then POSTs
        to the relayer with relayer API key auth (no gas needed).
        Returns transaction ID on success, None on failure.
        """
        if not self._clob:
            return None
        self._ensure_http()
        try:
            from eth_abi import encode
            from py_builder_relayer_client.signer import Signer as RelaySigner
            from py_builder_relayer_client.config import get_contract_config
            from py_builder_relayer_client.builder.safe import build_safe_transaction_request
            from py_builder_relayer_client.models import (
                SafeTransaction, SafeTransactionArgs, OperationType,
            )

            CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
            COLLATERAL = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

            private_key = self._clob.signer.private_key
            if not private_key:
                logger.error("No private key for redemption")
                return None

            # Encode redeemPositions(address, bytes32, bytes32, uint256[])
            # Function selector: 0x01a1ecbc (keccak of the signature)
            from web3 import Web3
            cond_bytes = bytes.fromhex(condition_id.replace("0x", "")).rjust(32, b"\x00")
            calldata = Web3.keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
            calldata += encode(
                ["address", "bytes32", "bytes32", "uint256[]"],
                [
                    Web3.to_checksum_address(COLLATERAL),
                    b"\x00" * 32,  # parentCollectionId (root)
                    cond_bytes,
                    [1, 2],  # both outcome slots
                ],
            )
            data_hex = "0x" + calldata.hex()

            # Build Safe transaction via relayer SDK
            relay_signer = RelaySigner(private_key, CHAIN_ID)
            config = get_contract_config(CHAIN_ID)
            from_address = relay_signer.address()

            # Get nonce from relayer
            nonce_resp = await self._http.get(
                f"https://relayer-v2.polymarket.com/nonce",
                params={"address": from_address, "type": "SAFE"},
            )
            nonce_resp.raise_for_status()
            nonce = nonce_resp.json().get("nonce", "0")

            safe_tx = SafeTransaction(
                to=Web3.to_checksum_address(CTF_ADDRESS),
                operation=OperationType.Call,
                data=data_hex,
                value="0",
            )

            safe_args = SafeTransactionArgs(
                from_address=from_address,
                nonce=nonce,
                chain_id=CHAIN_ID,
                transactions=[safe_tx],
            )

            txn_request = build_safe_transaction_request(
                signer=relay_signer,
                args=safe_args,
                config=config,
                metadata="Redeem winning tokens",
            ).to_dict()

            # POST to relayer with retries for rate limiting
            import asyncio as _aio
            result = None
            for attempt in range(3):
                resp = await self._http.post(
                    "https://relayer-v2.polymarket.com/submit",
                    headers={
                        "RELAYER_API_KEY": self._relayer_api_key,
                        "RELAYER_API_KEY_ADDRESS": from_address,
                    },
                    json=txn_request,
                )
                if resp.status_code == 429:
                    wait = (attempt + 1) * 5  # 5s, 10s, 15s
                    logger.warning("Relayer 429, waiting %ds...", wait)
                    await _aio.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()
                break
            if result is None:
                logger.error("Relayer still 429 after retries for %s", condition_id[:16])
                return None

            tx_id = result.get("transactionID", "")
            if not tx_id:
                logger.error("Relayer returned no transactionID for %s", condition_id[:16])
                return None

            # Poll for on-chain execution (relayer processes async)
            logger.info("Redeem submitted %s... txID=%s — polling for confirmation...",
                        condition_id[:16], tx_id)
            for poll in range(15):  # max ~30 seconds
                await _aio.sleep(2)
                try:
                    check = await self._http.get(
                        "https://relayer-v2.polymarket.com/transaction",
                        params={"id": tx_id},
                    )
                    if check.status_code != 200:
                        continue
                    txns = check.json()
                    if txns and isinstance(txns, list) and len(txns) > 0:
                        state = txns[0].get("state", "")
                        if state in ("STATE_MINED", "STATE_CONFIRMED"):
                            tx_hash = txns[0].get("transactionHash", "")
                            logger.info("Redeem CONFIRMED: %s (hash=%s)", tx_id, tx_hash[:16] if tx_hash else "?")
                            return tx_id
                        if state == "STATE_FAILED":
                            tx_hash = txns[0].get("transactionHash", "")
                            logger.error("Redeem FAILED on-chain: %s (hash=%s)", tx_id, tx_hash[:16] if tx_hash else "?")
                            return None
                        # Still processing (STATE_NEW, STATE_EXECUTED, etc)
                        if poll % 5 == 4:
                            logger.info("Redeem still pending: %s state=%s", tx_id, state)
                except Exception as poll_err:
                    logger.warning("Poll error: %s", poll_err)

            logger.warning("Redeem timeout (no confirmation after 30s): %s", tx_id)
            return None

        except Exception as e:
            logger.error("redeem_positions failed: %s", e)
            if "closed" in str(e).lower():
                self._ensure_http()
            return None

    async def get_condition_id(self, slug: str) -> Optional[str]:
        """Get the conditionId for a market slug from Gamma API."""
        try:
            resp = await self._http.get(
                f"{GAMMA_API}/events",
                params={"slug": slug, "limit": 1},
            )
            resp.raise_for_status()
            events = resp.json()
            if not events:
                return None
            markets = events[0].get("markets", [])
            if not markets:
                return None
            return markets[0].get("conditionId")
        except Exception as e:
            logger.debug("get_condition_id(%s) failed: %s", slug, e)
            return None

    async def get_redeemable_condition_ids(self, hours: int = 1) -> list[str]:
        """Get conditionIds of recent resolved 5-min BTC markets."""
        self._ensure_http()
        try:
            now = int(time.time())
            condition_ids = []

            async def _fetch_one(ts: int):
                slug = f"btc-updown-5m-{ts}"
                resp = await self._http.get(
                    f"{GAMMA_API}/events",
                    params={"slug": slug, "limit": 1},
                )
                if resp.status_code == 200:
                    events = resp.json()
                    if events:
                        for m in events[0].get("markets", []):
                            cid = m.get("conditionId")
                            if cid:
                                condition_ids.append(cid)

            windows = hours * 12  # 12 windows per hour
            all_ts = [((now - i * 300) // 300) * 300 for i in range(1, windows + 1)]
            # Process in batches of 10 to avoid overwhelming the HTTP client
            batch_size = 10
            for batch_start in range(0, len(all_ts), batch_size):
                batch = all_ts[batch_start:batch_start + batch_size]
                await asyncio.gather(*[_fetch_one(ts) for ts in batch])
            return condition_ids
        except Exception as e:
            logger.debug("get_redeemable_condition_ids failed: %s", e)
            return []

    # ── Info ──

    def time_remaining_s(self) -> int:
        """Seconds remaining in current 15-min window."""
        if not self._current_market:
            return 0
        return max(0, self._current_market.end_time - int(time.time()))

    def minutes_elapsed(self) -> float:
        """Minutes elapsed since window start."""
        if not self._current_market:
            return 0
        return (time.time() - self._current_market.window_start) / 60
