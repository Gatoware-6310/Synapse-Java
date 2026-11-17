package xyz.gatoware.synapse.transformer;

/** DROP discards excess routed assignments; ROUTE_ALL runs every selected assignment. */
public enum OverflowPolicy {
    DROP,
    ROUTE_ALL
}
