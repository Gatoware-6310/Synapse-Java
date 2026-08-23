import static org.junit.jupiter.api.Assertions.assertEquals;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import xyz.gatoware.synapse.image.Images;
import xyz.gatoware.synapse.matrix.Matrix;

public class ImagesTest {
	@Test
	void bufferedImageRoundTripsThroughRgbMatrix() {
		BufferedImage source = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
		source.setRGB(0, 0, 0xff0000);
		source.setRGB(1, 0, 0x00ff00);

		Matrix matrix = Images.fromBufferedImage(source);
		BufferedImage result = Images.toBufferedImage(matrix, 2, 1, 3);

		assertEquals(6, matrix.rows());
		assertEquals(1.0f, matrix.values[0][0], 1e-6f);
		assertEquals(1.0f, matrix.values[3][0], 1e-6f);
		assertEquals(0xff0000, result.getRGB(0, 0) & 0xffffff);
		assertEquals(0x00ff00, result.getRGB(1, 0) & 0xffffff);
	}

	@Test
	void saveAndLoadPreserveImagePixels() throws Exception {
		BufferedImage source = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
		source.setRGB(0, 0, 0x336699);
		Path file = Files.createTempFile("synapse-image", ".png");
		try {
			Images.save(Images.fromBufferedImage(source), 1, 1, 3, file);
			Matrix loaded = Images.load(file);
			BufferedImage result = Images.toBufferedImage(loaded, 1, 1, 3);
			assertEquals(0x336699, result.getRGB(0, 0) & 0xffffff);
		} finally {
			Files.deleteIfExists(file);
		}
	}
}
