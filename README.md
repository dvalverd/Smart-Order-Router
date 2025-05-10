 # SOR and Optimization Using Cont & Kukanov's Static Cost Model

This project implements and tunes a Smart Order Router (SOR) following the static cost model from Cont & Kukanov's paper _"Optimal Order Placement in Limit Order Markets"_. The router aims to minimize execution cost when buying 5,000 shares across multiple venues, using mocked market data over a 9-minute window.

## Approach

- **Allocator**: Implemented directly from `allocator_pseudocode.txt`, allocating shares across venues using the static cost model with parameters: `lambda_over`, `lambda_under`, and `theta_queue`.
- **Execution Engine**: Sequentially feeds Level-1 market snapshots (one per unique timestamp and venue), and attempts to execute the allocated quantities, filling up to the venue’s displayed ask size.
- **Unfilled Shares**: Any unexecuted quantity rolls forward to the next snapshot.
- **Grid Search**: Searches over a fixed grid of risk parameters to find the combination that minimizes total cash spent.

## Baselines

The SOR strategy is benchmarked against:
1. **Best Ask** – Always hitting the best available ask price.
2. **TWAP** – Time-weighted average price, splitting the order evenly across 9 equal time buckets.
3. **VWAP** – Volume-weighted average price, weighted by displayed ask size over all venues and timestamps.

## Files

- `backtest.py`: Main script. Loads and cleans data, runs the allocator and backtest, benchmarks against baselines, performs grid search, and outputs final JSON results.
- `results.png`: Cumulative cost plot for the best parameter set.
- `README.md`: This document.

## Parameter Grid

We use a small discrete grid search over:

- `lambda_over`: [0.01, 0.05, 0.1]
- `lambda_under`: [0.01, 0.05, 0.1]
- `theta_queue`: [0.1, 0.5, 1.0]

The combination that minimizes total cost is selected as the best configuration.

## Output

Upon completion, the script prints a JSON object containing:

- Best parameter set
- Total and average cost for the SOR strategy
- Baseline results (Best Ask, TWAP, VWAP)
- Savings vs each baseline in basis points
- A cumulative cost plot (`results.png`)

## Suggested Improvement

A realistic extension would be to model **queue position and slippage risk** more accurately. Currently, we assume immediate execution up to the displayed size, but in real markets, displayed liquidity may not be entirely available due to other order flow. Simulating queue depletion and incorporating fill probability would make the allocator more robust.

## Requirements

- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`
- No external data or services are used.

## Running the Script

```bash
python backtest.py
