Deterministic seed entry points:
- alns_min.py: function build_ws_round_robin_seed(...)
- main.py: no-seed flow calls build_ws_round_robin_seed for gamma==0 when full-coverage seed is infeasible.
Key log:
- [ALNS] no-seed deterministic ws-round-robin exact seed.
- [ALNS] ws-round-robin exact seed ready | cmax=...
