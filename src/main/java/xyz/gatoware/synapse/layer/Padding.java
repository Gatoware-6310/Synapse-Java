package xyz.gatoware.synapse.layer;

/** Padding modes for convolution layers. */
public enum Padding {
	/** Do not pad the input. */
	VALID,
	/** Pad the input so spatial output size is preserved for stride 1. */
	SAME
}
