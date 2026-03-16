#!/usr/bin/env python3
"""
Pattern trading bot v3 for Polymarket 5-min BTC UP/DOWN markets.

Strategy v3 (simple):
  - Buy UP when midpoint >= 80¢, buy DOWN when DOWN midpoint >= 80¢
  - Fixed $40 per trade
  - SL at 50¢ (midpoint-based, fast)
  - Entry window: 120s to 30s before market close
  - Autoclaim after winning trades (background retry)

Usage:
  python scripts/pattern_bot.py                # live trading
  python scripts/pattern_bot.py --dry-run      # simulate without orders
  python scripts/pattern_bot.py --derive-key   # derive private key from seed
"""

import argparse
import asyncio
import csv
import getpass
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from enum import Enum

# Add parent dir to path so we can import polymarket package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polymarket.client import PolyClient  # noqa: E402

# ── Constants ──

ENTRY_THRESHOLD = 0.87     # Buy when UP or DOWN midpoint >= this
ENTRY_CEILING = 0.93       # Don't buy above this price
SL_HIGH = 0.70             # SL (fixed for all entries)
SL_LOW = 0.70              # SL (fixed for all entries)
BET_SIZE = 40              # Fixed $40 per trade
MIN_FILL = 75              # Minimum fill to keep trade ($75)
ENTRY_WINDOW_START = 60    # Start evaluating at 60s remaining
ONLY_DOWN = False          # Trade BOTH sides (UP has better edge)
ENTRY_WINDOW_END = 30      # Stop evaluating at 30s remaining
MAX_CONSECUTIVE_SL = 2     # Pause after N consecutive SLs
SL_PAUSE_WINDOWS = 2       # Skip N windows (10 min) after max SLs
DAILY_STOP_LOSS = -200     # Stop trading if daily loss exceeds this
POLL_FAST = 0.2            # Poll interval during monitoring (fast SL)
POLL_NORMAL = 0.5          # Poll interval while waiting
FILL_TIMEOUT = 30          # Seconds to wait for order fill
TRADE_LOG = os.path.join(os.path.dirname(__file__), "trade_history.csv")

TRADE_CSV_FIELDS = [
    "timestamp", "market_slug", "side", "entry_price", "shares", "cost",
    "sl_price", "fill_price", "result", "pnl", "session_pnl",
    "trade_num", "win_rate", "window_start", "window_end",
    "entry_threshold", "entry_ceiling", "bet_size",
]


class State(Enum):
    IDLE = "IDLE"
    WAITING = "WAITING"
    ENTERING = "ENTERING"
    MONITORING = "MONITORING"
    AWAITING_RESOLUTION = "AWAITING"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pattern_bot")


class PatternBot:
    def __init__(self, client: PolyClient, dry_run: bool = False):
        self.client = client
        self.dry_run = dry_run
        self.state = State.IDLE
        self.last_window: int = 0
        self.position: dict | None = None
        self.total_pnl = 0.0
        self.trade_count = 0
        self.wins = 0
        self._running = True
        self._paused_no_cash = False
        self._last_redeem_sweep = 0.0
        self._redeemed_conditions: set = set()
        self._consecutive_sl = 0
        self._sl_pause_until = 0.0

    async def run(self):
        self._print_banner()
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                log.error("Tick error: %s", e)
            sleep = POLL_FAST if self.state == State.MONITORING else POLL_NORMAL
            await asyncio.sleep(sleep)

    async def _tick(self):
        # If paused (no cash), check balance every 30s
        if self._paused_no_cash:
            if not self.dry_run:
                balance = await self.client.get_balance()
                available = 0.0
                if balance:
                    bal_val = balance.get("balance", 0)
                    available = float(bal_val) if bal_val else 0.0
                if available >= BET_SIZE:
                    log.info("[RESUMED] Balance restored: $%.2f — resuming trading", available)
                    self._paused_no_cash = False
                else:
                    await asyncio.sleep(28)
                    return
            else:
                self._paused_no_cash = False

        market = await self.client.discover_current_market()
        if not market:
            return

        remaining = self.client.time_remaining_s()

        if self.state == State.IDLE:
            # Periodic sweep for unclaimed wins (every 5 min)
            if time.time() - self._last_redeem_sweep > 300:
                asyncio.ensure_future(self._redeem_sweep(hours=2))
            if market.window_start != self.last_window:
                self.last_window = market.window_start
                self.state = State.WAITING
                log.info("[IDLE] New window: %s (%ds remaining)", market.slug[-15:], remaining)

        elif self.state == State.WAITING:
            # Entry window: 120s to 30s before close
            if ENTRY_WINDOW_END < remaining <= ENTRY_WINDOW_START:
                await self._evaluate_entry(market, remaining)
            elif remaining <= ENTRY_WINDOW_END:
                log.info("[EVAL] Window closed (%ds left), skipping", remaining)
                self.state = State.IDLE

        elif self.state == State.MONITORING:
            await self._monitor_position(market, remaining)

        elif self.state == State.AWAITING_RESOLUTION:
            await self._check_resolution(market)

    # ── Entry (v3: simple threshold) ──

    async def _evaluate_entry(self, market, remaining):
        # Check SL pause
        if time.time() < self._sl_pause_until:
            remaining_pause = int(self._sl_pause_until - time.time())
            if remaining_pause % 30 == 0:  # Log every 30s
                log.info("[PAUSE] SL cooldown — %ds remaining", remaining_pause)
            return

        mid_up = await self.client.get_midpoint(market.token_up)
        if mid_up is None:
            return  # stay in WAITING, retry next tick

        mid_down = 1.0 - mid_up

        log.info("[EVAL] UP=%.2f¢ DOWN=%.2f¢ (%ds left)", mid_up * 100, mid_down * 100, remaining)

        # Trade both sides — prefer UP (historically more reliable)
        if ENTRY_THRESHOLD <= mid_up <= ENTRY_CEILING:
            side = "UP"
            token_id = market.token_up
            price = min(0.99, mid_up + 0.01)
        elif ENTRY_THRESHOLD <= mid_down <= ENTRY_CEILING:
            side = "DOWN"
            token_id = market.token_down
            price = min(0.99, mid_down + 0.01)
        else:
            return  # stay in WAITING, retry next tick

        # Check balance
        if not self.dry_run:
            balance = await self.client.get_balance()
            available = 0.0
            if balance:
                bal_val = balance.get("balance", 0)
                available = float(bal_val) if bal_val else 0.0
            if available < BET_SIZE:
                log.warning("[PAUSED] No cash — balance $%.2f < $%d. Bot paused.", available, BET_SIZE)
                self._paused_no_cash = True
                self.state = State.IDLE
                return

        # Liquidity check removed — FAK order handles partial fills
        shares = round(BET_SIZE / price, 1)
        if shares < 1:
            log.warning("[EVAL] SKIP — too few shares (%.1f)", shares)
            self.state = State.IDLE
            return

        sl = SL_HIGH if price >= 0.90 else SL_LOW
        log.info("[ENTRY] BUY %s @ %.2f¢ (%d shares, $%d) SL=%.0f¢",
                 side, price * 100, shares, BET_SIZE, sl * 100)

        order_id = None
        if not self.dry_run:
            result = await self.client.place_limit_buy(token_id, price, shares)
            if not result:
                log.error("[ENTRY] Order failed")
                self.state = State.IDLE
                return
            order_id = result.get("orderID") or result.get("id", "")
            log.info("[ENTRY] Order placed: %s", order_id)
        else:
            order_id = "DRY-RUN"

        self.position = {
            "token_id": token_id,
            "side": side,
            "entry_price": price,
            "shares": shares,
            "cost": BET_SIZE,
            "order_id": order_id,
            "market_slug": market.slug,
            "window_start": market.window_start,
            "filled": self.dry_run,
            "entry_time": time.time(),
            "sl_price": sl,
        }
        self.state = State.MONITORING

    # ── Monitoring (simple SL at 50¢) ──

    async def _monitor_position(self, market, remaining):
        if not self.position:
            self.state = State.IDLE
            return

        # Check fill
        if not self.position["filled"]:
            await self._check_fill()
            if not self.position["filled"]:
                if time.time() - self.position["entry_time"] > FILL_TIMEOUT:
                    log.warning("[MONITOR] Order not filled after %ds, cancelling", FILL_TIMEOUT)
                    if not self.dry_run:
                        await self.client.cancel_order(self.position["order_id"])
                    self.position = None
                    self.state = State.IDLE
                return
            # Pre-approve allowance for fast SL sell
            if not self.dry_run:
                try:
                    await self.client.ensure_allowance(token_id=self.position["token_id"])
                except Exception as e:
                    log.warning("[MONITOR] Allowance pre-approval failed: %s", e)

        # Window ended
        if market.window_start != self.position["window_start"] or remaining <= 0:
            log.info("[MONITOR] Window ended, awaiting resolution")
            self.state = State.AWAITING_RESOLUTION
            return

        # Check midpoint for SL
        mid = await self.client.get_midpoint(self.position["token_id"])
        sl_price = self.position.get("sl_price", SL_LOW)
        if mid is not None:
            if mid <= sl_price:
                log.warning("[SL] TRIGGERED! %s @ %.2f¢ <= %.0f¢ — selling NOW",
                            self.position["side"], mid * 100, sl_price * 100)
                sl_attempts = self.position.get("sl_sell_attempts", 0)
                # Get REAL on-chain balance (may differ from fill size due to fees)
                real_balance = None
                try:
                    cond_bal = await self.client.get_conditional_balance(self.position["token_id"])
                    if cond_bal:
                        real_balance = float(cond_bal) / 1e6  # 6 decimals like USDC
                        log.info("[SL] On-chain balance: %.2f tokens (fill was %.1f)",
                                 real_balance, self.position["shares"])
                except Exception as e:
                    log.warning("[SL] Balance check failed: %s", e)
                # Use real balance if available, otherwise fall back to fill size
                sell_size = real_balance if real_balance and real_balance > 0 else self.position["shares"]
                # Only refresh allowance on retries (pre-approved on fill)
                if sl_attempts > 0:
                    try:
                        await self.client.ensure_allowance(token_id=self.position["token_id"])
                        log.info("[SL] Allowance re-refreshed (attempt %d)", sl_attempts + 1)
                    except Exception as e:
                        log.warning("[SL] Allowance refresh failed: %s", e)
                # Try market sell (FOK taker order) — fills immediately against bids
                sell_price = max(mid - 0.03, 0.01)
                result = await self.client.place_market_sell(
                    self.position["token_id"], sell_size, sell_price,
                )
                if not result:
                    # Fallback: try limit sell (GTC maker order)
                    log.warning("[SL] Market sell failed, trying limit sell fallback")
                    result = await self.client.place_limit_sell(
                        self.position["token_id"], sell_price, sell_size,
                    )
                if result:
                    size_matched = float(result.get("takingAmount", 0) or 0)
                    recovery = size_matched if size_matched > 0 else mid * self.position["shares"]
                    pnl = recovery - self.position["cost"]
                    log.warning("[SL] SOLD! PnL: $%.2f", pnl)
                    self.position["sl_triggered"] = True
                    self.position["sl_price"] = mid
                    self.state = State.AWAITING_RESOLUTION
                    return
                else:
                    self.position["sl_sell_attempts"] = sl_attempts + 1
                    if sl_attempts < 2:
                        log.error("[SL] Sell failed (attempt %d) — will wait 2s + retry", sl_attempts + 1)
                        await asyncio.sleep(2)  # extra wait for allowance propagation
                    else:
                        log.error("[SL] Sell failed (attempt %d) — retry next cycle", sl_attempts + 1)

            # Log every 10s
            if int(time.time()) % 10 == 0:
                log.info("[MONITOR] %s @ %.2f¢ (entry: %.2f¢, SL: %.0f¢, %ds left)",
                         self.position["side"], mid * 100,
                         self.position["entry_price"] * 100,
                         sl_price * 100, remaining)

    async def _check_fill(self):
        if self.dry_run or not self.position:
            return
        details = await self.client.get_order_details(self.position["order_id"])
        if not details:
            return
        size_matched = float(details.get("size_matched", 0) or details.get("sizeMatched", 0) or 0)
        if size_matched > 0:
            fill_price = float(details.get("price", self.position["entry_price"]))
            cost = size_matched * fill_price
            # Small fill — keep position, let it ride with SL protection
            if cost < MIN_FILL:
                log.info("[FILL] Partial fill $%.1f < $%d target — keeping position", cost, MIN_FILL)
            self.position["filled"] = True
            self.position["shares"] = size_matched
            self.position["cost"] = cost
            log.info("[FILL] Order filled: %.1f shares @ %.2f¢ ($%.1f)", size_matched, fill_price * 100, cost)

    # ── Resolution ──

    async def _check_resolution(self, market):
        if not self.position:
            self.state = State.IDLE
            return

        # For SL-triggered positions, calculate PnL immediately
        if self.position.get("sl_triggered"):
            sl_price = self.position.get("sl_price", 0)
            recovery = sl_price * self.position["shares"]
            pnl = recovery - self.position["cost"]
            self._record_trade(pnl, "STOP LOSS")
            return

        # Check market resolution via Gamma API
        resolution = await self.client.get_market_resolution(self.position["market_slug"])
        if resolution is None:
            mid = await self.client.get_midpoint(self.position["token_id"])
            if mid is not None:
                if mid >= 0.95:
                    resolution = self.position["side"]
                elif mid <= 0.05:
                    resolution = "DOWN" if self.position["side"] == "UP" else "UP"
            if resolution is None:
                elapsed = time.time() - (self.position.get("entry_time", 0))
                if elapsed > 120:
                    log.warning("[RESOLVE] Timeout — assuming %s based on entry signal", self.position["side"])
                    resolution = self.position["side"]
                else:
                    return

        won = (self.position["side"] == resolution)
        slug = self.position["market_slug"]
        if won:
            pnl = self.position["shares"] * 1.0 - self.position["cost"]
            self._record_trade(pnl, f"WIN ({resolution})")
            await self._redeem_winning(slug)
        else:
            pnl = -self.position["cost"]
            self._record_trade(pnl, f"LOSS ({resolution})")

    async def _redeem_winning(self, market_slug: str):
        """Auto-claim winning CTF tokens for USDC after a delay."""
        if self.dry_run:
            log.info("[REDEEM] DRY RUN — would redeem positions for %s", market_slug[-15:])
            return
        # Schedule redeem in background with delay (market needs time to settle on-chain)
        asyncio.ensure_future(self._redeem_with_retry(market_slug))

    async def _redeem_with_retry(self, market_slug: str):
        """Redeem with delay and retries — runs in background.

        Claim availability can take minutes after resolution, so we retry
        with increasing delays: 30s, 60s, 120s, 180s, 300s (total ~11.5 min).
        """
        delays = [30, 60, 120, 180, 300]  # seconds before each attempt
        for attempt, wait in enumerate(delays, 1):
            log.info("[REDEEM] Waiting %ds before claim attempt %d/%d for %s",
                     wait, attempt, len(delays), market_slug[-15:])
            await asyncio.sleep(wait)
            try:
                self.client._ensure_http()
                condition_id = await self.client.get_condition_id(market_slug)
                if not condition_id:
                    log.warning("[REDEEM] Could not get conditionId for %s (attempt %d)", market_slug[-15:], attempt)
                    continue
                tx_hash = await self.client.redeem_positions(condition_id)
                if tx_hash:
                    log.info("[REDEEM] ✓ Claimed! tx=%s for %s", tx_hash[:16], market_slug[-15:])
                    return
                log.warning("[REDEEM] Attempt %d/%d failed for %s (will retry)", attempt, len(delays), market_slug[-15:])
            except Exception as e:
                log.error("[REDEEM] Attempt %d/%d error: %s", attempt, len(delays), e)
        log.error("[REDEEM] All %d attempts failed for %s — may need manual claim", len(delays), market_slug[-15:])

    async def _redeem_sweep(self, hours: int = 1):
        """Sweep all redeemable positions from the last N hours."""
        self._last_redeem_sweep = time.time()
        try:
            condition_ids = await self.client.get_redeemable_condition_ids(hours=hours)
            to_redeem = [c for c in condition_ids if c not in self._redeemed_conditions]
            if not to_redeem:
                return
            log.info("[SWEEP] Found %d unredeemed positions (last %dh), attempting...", len(to_redeem), hours)
            redeemed = 0
            failed = 0
            for cid in to_redeem:
                tx_id = await self.client.redeem_positions(cid)
                if tx_id:
                    self._redeemed_conditions.add(cid)
                    redeemed += 1
                else:
                    failed += 1
                    # Don't add to _redeemed_conditions — retry next sweep
                await asyncio.sleep(3)  # spacing between redeems to avoid rate limits
            log.info("[SWEEP] Redeemed %d/%d positions (%d failed)",
                     redeemed, len(to_redeem), failed)
        except Exception as e:
            log.error("[SWEEP] Error: %s", e)

    def _record_trade(self, pnl: float, reason: str):
        self.total_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.wins += 1

        symbol = "+" if pnl >= 0 else ""
        win_rate = (self.wins / self.trade_count * 100) if self.trade_count else 0
        log.info("[RESULT] %s | PnL: %s$%.2f | Session: %s$%.2f (%d trades, %.0f%% WR)",
                 reason, symbol, pnl, "+" if self.total_pnl >= 0 else "",
                 self.total_pnl, self.trade_count, win_rate)

        # Write to CSV history
        try:
            file_exists = os.path.exists(TRADE_LOG)
            with open(TRADE_LOG, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=TRADE_CSV_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "market_slug": self.position.get("market_slug", "") if self.position else "",
                    "side": self.position.get("side", "") if self.position else "",
                    "entry_price": self.position.get("entry_price", 0) if self.position else 0,
                    "shares": self.position.get("shares", 0) if self.position else 0,
                    "cost": self.position.get("cost", 0) if self.position else 0,
                    "sl_price": self.position.get("sl_price", 0) if self.position else 0,
                    "fill_price": self.position.get("fill_price", self.position.get("entry_price", 0)) if self.position else 0,
                    "result": reason,
                    "pnl": round(pnl, 2),
                    "session_pnl": round(self.total_pnl, 2),
                    "trade_num": self.trade_count,
                    "win_rate": round(win_rate, 1),
                    "window_start": ENTRY_WINDOW_START,
                    "window_end": ENTRY_WINDOW_END,
                    "entry_threshold": ENTRY_THRESHOLD,
                    "entry_ceiling": ENTRY_CEILING,
                    "bet_size": BET_SIZE,
                })
        except Exception as e:
            log.warning("[CSV] Failed to write trade: %s", e)

        # Track consecutive SLs for pause logic
        if "STOP LOSS" in reason:
            self._consecutive_sl += 1
            if self._consecutive_sl >= MAX_CONSECUTIVE_SL:
                pause_secs = SL_PAUSE_WINDOWS * 300  # 5 min per window
                self._sl_pause_until = time.time() + pause_secs
                log.warning("[PAUSE] %d consecutive SLs — pausing %d min",
                            self._consecutive_sl, pause_secs // 60)
        else:
            self._consecutive_sl = 0

        # Daily stop loss check
        if self.total_pnl <= DAILY_STOP_LOSS:
            log.warning("[DAILY STOP] Session PnL $%.2f <= $%d — stopping bot",
                        self.total_pnl, DAILY_STOP_LOSS)
            self._running = False

        self.position = None
        self.state = State.IDLE

    # ── Lifecycle ──

    async def shutdown(self):
        log.info("Shutting down...")
        self._running = False
        if not self.dry_run:
            await self.client.cancel_all()
        win_rate = (self.wins / self.trade_count * 100) if self.trade_count else 0
        log.info("Final: %d trades | %d wins (%.0f%%) | PnL: $%.2f",
                 self.trade_count, self.wins, win_rate, self.total_pnl)
        await self.client.disconnect()

    def _print_banner(self):
        mode = "DRY RUN" if self.dry_run else "LIVE"
        log.info("=" * 60)
        log.info("  PATTERN BOT v3 [%s]", mode)
        log.info("  Entry: UP/DOWN >= %.0f¢ <= %.0f¢ | $%d per trade | SL: %.0f¢ (>=90¢) / %.0f¢ (80-90¢)",
                 ENTRY_THRESHOLD * 100, ENTRY_CEILING * 100, BET_SIZE, SL_HIGH * 100, SL_LOW * 100)
        log.info("  Window: %ds to %ds before close",
                 ENTRY_WINDOW_START, ENTRY_WINDOW_END)
        log.info("  Poll: %.1fs (monitor) / %.1fs (wait)", POLL_FAST, POLL_NORMAL)
        log.info("=" * 60)


def derive_key():
    """Derive private key from seed phrase using BIP-44."""
    try:
        from eth_account import Account
    except ImportError:
        print("Install eth-account: pip install eth-account")
        sys.exit(1)

    Account.enable_unaudited_hdwallet_features()
    phrase = getpass.getpass("Enter seed phrase (hidden): ")
    if not phrase.strip():
        print("Empty seed phrase")
        sys.exit(1)

    acct = Account.from_mnemonic(phrase.strip())
    print(f"\nAddress:     {acct.address}")
    print(f"Private key: {acct.key.hex()}")
    print(f"\nAdd to .env: POLY_PRIVATE_KEY={acct.key.hex()}")


async def main(dry_run: bool):
    try:
        from dotenv import load_dotenv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for rel in ["..", "../.."]:
            p = os.path.join(script_dir, rel, ".env")
            if os.path.exists(p):
                load_dotenv(p)
                break
    except ImportError:
        pass  # Railway: env vars set directly, no .env needed

    private_key = os.getenv("POLY_PRIVATE_KEY", "")
    funder = os.getenv("POLY_FUNDER_ADDRESS", "")
    api_key = os.getenv("POLY_API_KEY", "")
    relayer_key = os.getenv("POLY_RELAYER_API_KEY", "")

    if not private_key and not dry_run:
        log.error("POLY_PRIVATE_KEY not set (check env vars or .env file)")
        log.error("Run: python scripts/pattern_bot.py --derive-key")
        sys.exit(1)

    client = PolyClient()
    await client.connect(
        private_key=private_key,
        funder_address=funder,
        api_key=api_key,
        relayer_api_key=relayer_key,
    )

    bot = PatternBot(client, dry_run=dry_run)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(bot.shutdown()))

    await bot.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pattern trading bot v3 for Polymarket")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without placing orders")
    parser.add_argument("--derive-key", action="store_true", help="Derive private key from seed phrase")
    args = parser.parse_args()

    if args.derive_key:
        derive_key()
    else:
        asyncio.run(main(dry_run=args.dry_run))
