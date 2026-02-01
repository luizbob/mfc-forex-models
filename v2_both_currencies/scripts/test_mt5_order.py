"""
Quick test to check if MT5 can open and close trades.
"""
import MetaTrader5 as mt5
import time

SYMBOL = "EURUSDm"
MAGIC = 999999  # Test magic number
LOT = 0.01

def main():
    print("Initializing MT5...")
    if not mt5.initialize():
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return

    print(f"MT5 connected: {mt5.account_info().server}")
    print(f"Account: {mt5.account_info().login}")

    # Check symbol
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        mt5.symbol_select(SYMBOL, True)
        info = mt5.symbol_info(SYMBOL)

    if info is None:
        print(f"Symbol {SYMBOL} not found")
        return

    print(f"Symbol: {SYMBOL}, Bid: {info.bid}, Ask: {info.ask}")
    print(f"Trade mode: {info.trade_mode} (0=disabled, 4=full)")

    # Try to open a BUY
    print("\n--- Opening BUY ---")
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print(f"symbol_info_tick returned None: {mt5.last_error()}")
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 10,
        "magic": MAGIC,
        "comment": "Test order",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print(f"Request: {request}")
    result = mt5.order_send(request)

    if result is None:
        print(f"order_send returned None: {mt5.last_error()}")
        mt5.shutdown()
        return

    print(f"Result: retcode={result.retcode}, comment={result.comment}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Open failed!")
        mt5.shutdown()
        return

    print(f"Opened ticket: {result.order}")

    # Wait a bit
    print("\nWaiting 3 seconds...")
    time.sleep(3)

    # Try to close
    print("\n--- Closing position ---")
    positions = mt5.positions_get(symbol=SYMBOL)
    print(f"Positions found: {len(positions) if positions else 0}")

    for pos in positions or []:
        if pos.magic != MAGIC:
            continue

        print(f"Position: ticket={pos.ticket}, type={pos.type}, profit={pos.profit}")

        tick = mt5.symbol_info_tick(SYMBOL)
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL,
            "position": pos.ticket,
            "price": tick.bid,
            "deviation": 10,
            "magic": MAGIC,
            "comment": "Close test",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"Close request: {close_request}")
        close_result = mt5.order_send(close_request)

        if close_result is None:
            print(f"Close order_send returned None: {mt5.last_error()}")
        else:
            print(f"Close result: retcode={close_result.retcode}, comment={close_result.comment}")

    mt5.shutdown()
    print("\nDone!")

if __name__ == "__main__":
    main()
