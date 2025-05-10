import pandas as pd
import numpy as np
import json
from itertools import product
import matplotlib.pyplot as plt


TOTAL_ORDER_SIZE = 5000
PARAM_GRID = {
    'lambda_over': [0.01, 0.05, 0.1],
    'lambda_under': [0.01, 0.05, 0.1],
    'theta_queue': [0.1, 0.5, 1.0]
}


def allocator(venues, total_qty, lambda_over, lambda_under, theta_queue):
    bids = []
    for venue_id, price, size in venues:
        if size == 0:
            continue
        q_hat = size
        cost = price + lambda_over * max(0, -q_hat) + lambda_under * max(0, q_hat - size) + theta_queue * q_hat
        bids.append((venue_id, cost, size))
    if not bids:
        return []
    total_cost = sum([1.0 / b[1] for b in bids])
    allocations = [(b[0], int(round((1.0 / b[1]) / total_cost * total_qty))) for b in bids]
    return allocations


df = pd.read_csv("l1_day.csv")
df = df.sort_values(by=["ts_event", "publisher_id"])
df = df.drop_duplicates(subset=["ts_event", "publisher_id"], keep="first")


snapshots = list(df.groupby("ts_event"))


def run_backtest(lambda_over, lambda_under, theta_queue):
    filled = 0
    total_cash = 0
    cum_costs = []
    remaining = TOTAL_ORDER_SIZE

    for ts, snap in snapshots:
        venues = []
        for _, row in snap.iterrows():
            venues.append((row['publisher_id'], row['ask_px_00'], row['ask_sz_00']))
        allocs = allocator(venues, remaining, lambda_over, lambda_under, theta_queue)
        for venue_id, alloc_qty in allocs:
            venue_row = snap[snap['publisher_id'] == venue_id].iloc[0]
            px = venue_row['ask_px_00']
            sz = venue_row['ask_sz_00']
            fill = min(alloc_qty, sz, remaining)
            total_cash += fill * px
            filled += fill
            remaining -= fill
            if remaining <= 0:
                break
        cum_costs.append(total_cash)
        if remaining <= 0:
            break

    avg_price = total_cash / filled if filled > 0 else float('inf')
    return total_cash, avg_price, cum_costs


def best_ask_strategy():
    filled = 0
    total_cash = 0
    remaining = TOTAL_ORDER_SIZE
    for ts, snap in snapshots:
        snap = snap[snap['ask_sz_00'] > 0]
        if snap.empty:
            continue
        best_row = snap.loc[snap['ask_px_00'].idxmin()]
        fill = min(remaining, best_row['ask_sz_00'])
        total_cash += fill * best_row['ask_px_00']
        filled += fill
        remaining -= fill
        if remaining <= 0:
            break
    avg_price = total_cash / filled
    return total_cash, avg_price


def twap_strategy():
    time_buckets = 9
    shares_per_bucket = TOTAL_ORDER_SIZE // time_buckets
    total_cash = 0
    filled = 0

    bucket_size = len(snapshots) // time_buckets
    buckets = [snapshots[i * bucket_size: (i + 1) * bucket_size] for i in range(time_buckets)]
    if time_buckets * bucket_size < len(snapshots):
        buckets[-1].extend(snapshots[time_buckets * bucket_size:])

    for bucket in buckets:
        remaining = shares_per_bucket
        for ts, snap in bucket:
            snap = snap[snap['ask_sz_00'] > 0]
            if snap.empty:
                continue
            best_row = snap.loc[snap['ask_px_00'].idxmin()]
            fill = min(remaining, best_row['ask_sz_00'])
            total_cash += fill * best_row['ask_px_00']
            filled += fill
            remaining -= fill
            if remaining <= 0:
                break

    avg_price = total_cash / filled
    return total_cash, avg_price


def vwap_strategy():
    df_valid = df[df['ask_sz_00'] > 0]
    vwap_price = np.sum(df_valid['ask_px_00'] * df_valid['ask_sz_00']) / np.sum(df_valid['ask_sz_00'])
    total_cash = vwap_price * TOTAL_ORDER_SIZE
    avg_price = vwap_price
    return total_cash, avg_price


results = []
for lo, lu, tq in product(PARAM_GRID['lambda_over'], PARAM_GRID['lambda_under'], PARAM_GRID['theta_queue']):
    total_cash, avg_price, cum_costs = run_backtest(lo, lu, tq)
    results.append({
        "lambda_over": lo,
        "lambda_under": lu,
        "theta_queue": tq,
        "total_cash": total_cash,
        "avg_price": avg_price,
        "cum_costs": cum_costs
    })

best = min(results, key=lambda x: x["total_cash"])


best_ask_cash, best_ask_avg = best_ask_strategy()
twap_cash, twap_avg = twap_strategy()
vwap_cash, vwap_avg = vwap_strategy()


savings_vs_best_ask = (1 - best["avg_price"] / best_ask_avg) * 10000
savings_vs_twap = (1 - best["avg_price"] / twap_avg) * 10000
savings_vs_vwap = (1 - best["avg_price"] / vwap_avg) * 10000

output = {
    "best_params": {
        "lambda_over": best["lambda_over"],
        "lambda_under": best["lambda_under"],
        "theta_queue": best["theta_queue"]
    },
    "router_total_cash": best["total_cash"],
    "router_avg_price": best["avg_price"],
    "best_ask_total_cash": best_ask_cash,
    "best_ask_avg_price": best_ask_avg,
    "twap_total_cash": twap_cash,
    "twap_avg_price": twap_avg,
    "vwap_total_cash": vwap_cash,
    "vwap_avg_price": vwap_avg,
    "savings_vs_best_ask_bps": savings_vs_best_ask,
    "savings_vs_twap_bps": savings_vs_twap,
    "savings_vs_vwap_bps": savings_vs_vwap
}

print(json.dumps(output, indent=2))


plt.plot(best["cum_costs"])
plt.title("Cumulative Cost Over Time (Best Router)")
plt.xlabel("Snapshot Index")
plt.ylabel("Cumulative Cost ($)")
plt.grid()
plt.savefig("results.png")
