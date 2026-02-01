"""
Monitor MFC data file for bad values.
Run alongside the trader to catch data issues in real-time.
"""
import json
import time
from pathlib import Path
from datetime import datetime
import os

MFC_FILE = Path(os.environ.get('APPDATA', '')) / "MetaQuotes/Terminal/Common/Files/DWX/DWX_MFC_Auto.txt"

# Thresholds
MAX_MFC_VALUE = 2.5
MAX_JUMP = 0.8  # Alert if value jumps more than this between consecutive bars

def check_mfc_data():
    """Check the MFC file for anomalies."""
    if not MFC_FILE.exists():
        return None, "File not found"

    try:
        with open(MFC_FILE, 'r') as f:
            data = json.loads(f.read().strip())
    except Exception as e:
        return None, f"Parse error: {e}"

    issues = []

    for tf_key, currencies in data.items():
        if not isinstance(currencies, dict):
            continue

        for currency, bars in currencies.items():
            if not isinstance(bars, dict):
                continue

            # Sort by datetime
            sorted_bars = sorted(bars.items(), key=lambda x: x[0])

            prev_value = None
            for bar_time, value in sorted_bars:
                # Range check
                if value > MAX_MFC_VALUE or value < -MAX_MFC_VALUE:
                    issues.append(f"OUT OF RANGE: {currency} {tf_key} @ {bar_time}: {value:.4f}")

                # Jump check
                if prev_value is not None:
                    jump = abs(value - prev_value)
                    if jump > MAX_JUMP:
                        issues.append(f"SPIKE: {currency} {tf_key} @ {bar_time}: {prev_value:.4f} -> {value:.4f} (jump={jump:.4f})")

                prev_value = value

    return data, issues


def main():
    print("=" * 60)
    print("MFC DATA MONITOR")
    print("=" * 60)
    print(f"Watching: {MFC_FILE}")
    print(f"Max MFC value: +/-{MAX_MFC_VALUE}")
    print(f"Max jump between bars: {MAX_JUMP}")
    print("=" * 60)
    print()

    last_mtime = 0
    check_count = 0
    issue_count = 0

    while True:
        try:
            if MFC_FILE.exists():
                current_mtime = MFC_FILE.stat().st_mtime

                if current_mtime != last_mtime:
                    last_mtime = current_mtime
                    check_count += 1

                    data, issues = check_mfc_data()

                    timestamp = datetime.now().strftime("%H:%M:%S")

                    if issues:
                        issue_count += len(issues)
                        print(f"\n[{timestamp}] CHECK #{check_count} - ISSUES FOUND:")
                        for issue in issues:
                            print(f"  ! {issue}")
                    else:
                        # Show brief status every check
                        print(f"[{timestamp}] CHECK #{check_count} - OK (total issues: {issue_count})")

            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\nStopped.")
            print(f"Total checks: {check_count}")
            print(f"Total issues found: {issue_count}")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
