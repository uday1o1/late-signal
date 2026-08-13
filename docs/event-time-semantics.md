# Event-time semantics

LateSignal predicts at click time and treats truth as a separate event stream.
A click record contains an identifier, event time, and click-time features, but it never contains its eventual label or availability time.
Only the label oracle and final evaluator hold complete truth records.

The simulator advances through fixed hourly boundaries.
At boundary `t`, it performs these operations in order:

1. Predict clicks in `(previous_boundary, t]` with the current model.
2. Atomically persist the prediction ledger and seal it when the final click is included.
3. Deliver the clicks to the delayed-label method.
4. Request positive reveals and negative maturities whose availability time is at most `t`.
5. Append emitted records after enforcing `available_at <= t`.
6. Make an update-credit decision at aligned daily boundaries before the last click time.
7. Train only from the legal availability ledger.
8. Atomically checkpoint all mutable simulation state.

The prediction step therefore precedes a reveal at the same timestamp.
Golden tests cover one second before, exactly at, and one second after positive reveal and negative maturity boundaries.

The update horizon ends at the last click time.
The clock continues through the first hourly boundary at or after the last truth availability time without spending later credits.
The evaluator reads final labels only after the prediction ledger is sealed and the oracle is fully drained.

Every daily checkpoint contains the model, method, scheduler, oracle, ledgers, click cursor, clock boundaries, compute counters, event trace, seed, and random-generator state.
Resume reconstructs the authored configuration and synthetic fixture, verifies both hashes, restores state, and writes to a new output directory.
On the same software stack, uninterrupted and resumed runs must produce identical prediction, availability, credit, exposure, and event-ledger hashes.
