package xyz.gatoware.synapse.layer;

/** Padding modes for convolution layers. */
public enum Padding {
	/** Do not pad the input. */
	NONE,
	/** Pad the input so spatial output size is preserved for stride 1. */
	SAME;

	/** @deprecated Use {@link #NONE}. */
	@Deprecated
	public static final Padding VALID = NONE;
}
