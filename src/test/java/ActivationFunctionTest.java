import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.activation.ELU;
import xyz.gatoware.synapse.activation.GELU;
import xyz.gatoware.synapse.activation.LeakyReLU;
import xyz.gatoware.synapse.activation.ReLU;
import xyz.gatoware.synapse.activation.SiLU;
import xyz.gatoware.synapse.activation.Sigmoid;
import xyz.gatoware.synapse.activation.Softmax;
import xyz.gatoware.synapse.activation.Swish;
import xyz.gatoware.synapse.activation.Tanh;

public class ActivationFunctionTest {
	private static final float EPSILON = 0.0001f;

	@Test void reluClampsNegativeValues() { assertEquals(0, new ReLU().apply(-2), EPSILON); assertEquals(2, new ReLU().apply(2), EPSILON); }
	@Test void sigmoidHasExpectedValues() { assertEquals(0.5, new Sigmoid().apply(0), EPSILON); assertEquals(0.880797, new Sigmoid().apply(2), EPSILON); }
	@Test void tanhHasExpectedValues() { assertEquals(0, new Tanh().apply(0), EPSILON); assertEquals((float) Math.tanh(1), new Tanh().apply(1), EPSILON); }
	@Test void leakyReluUsesDefaultAndCustomSlope() { assertEquals(-0.02, new LeakyReLU().apply(-2), EPSILON); assertEquals(-0.4, new LeakyReLU(0.2f).apply(-2), EPSILON); }
	@Test void geluHasExpectedValues() { assertEquals(0, new GELU().apply(0), EPSILON); assertEquals(0.8413447, new GELU().apply(1), EPSILON); }
	@Test void siluAndSwishAreEquivalent() { assertEquals(new SiLU().apply(2), new Swish().apply(2), EPSILON); }
	@Test void eluUsesDefaultAndCustomAlpha() { assertEquals((float) Math.expm1(-1), new ELU().apply(-1), EPSILON); assertEquals(2 * (float) Math.expm1(-1), new ELU(2).apply(-1), EPSILON); }
	@Test void softmaxNormalizesASequence() { assertArrayEquals(new float[] { 0.2689414f, 0.7310586f }, new Softmax().apply(new float[] { 1, 2 }), EPSILON); assertEquals(1, new Softmax().apply(3), EPSILON); }
	@Test void parametersMustBeValid() { assertThrows(IllegalArgumentException.class, () -> new LeakyReLU(-1)); assertThrows(IllegalArgumentException.class, () -> new ELU(0)); }
}
