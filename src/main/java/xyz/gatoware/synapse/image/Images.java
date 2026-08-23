package xyz.gatoware.synapse.image;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

import javax.imageio.ImageIO;

import xyz.gatoware.synapse.matrix.Matrix;

/** Utilities for converting image files and BufferedImages to Synapse matrices. */
public final class Images {
	private Images() {
	}

	/** Loads an image as a normalized RGB matrix. */
	public static Matrix load(String filename) throws IOException {
		return load(Path.of(filename));
	}

	/** Loads an image as a normalized RGB matrix. */
	public static Matrix load(Path path) throws IOException {
		BufferedImage image = ImageIO.read(path.toFile());
		if (image == null)
			throw new IOException("Unsupported or invalid image: " + path);
		return fromBufferedImage(image);
	}

	/** Loads and resizes an image, returning a normalized RGB matrix. */
	public static Matrix load(String filename, int width, int height) throws IOException {
		return load(Path.of(filename), width, height);
	}

	/** Loads and resizes an image, returning a normalized RGB matrix. */
	public static Matrix load(Path path, int width, int height) throws IOException {
		if (width <= 0 || height <= 0)
			throw new IllegalArgumentException("Image dimensions must be positive");
		BufferedImage image = ImageIO.read(path.toFile());
		if (image == null)
			throw new IOException("Unsupported or invalid image: " + path);
		if (image.getWidth() != width || image.getHeight() != height)
			image = resize(image, width, height);
		return fromBufferedImage(image);
	}

	/** Converts a BufferedImage to a normalized RGB matrix in channel-first order. */
	public static Matrix fromBufferedImage(BufferedImage image) {
		if (image == null)
			throw new IllegalArgumentException("Image cannot be null");
		return fromBufferedImage(image, 3);
	}

	/** Converts a BufferedImage to a normalized one-channel or RGB matrix. */
	public static Matrix fromBufferedImage(BufferedImage image, int channels) {
		if (image == null)
			throw new IllegalArgumentException("Image cannot be null");
		validateChannels(channels);
		int width = image.getWidth();
		int height = image.getHeight();
		long flattened = (long) width * height * channels;
		if (flattened > Integer.MAX_VALUE)
			throw new IllegalArgumentException("Image is too large");
		Matrix matrix = new Matrix((int) flattened, 1);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rgb = image.getRGB(x, y);
				float red = ((rgb >>> 16) & 0xff) / 255.0f;
				float green = ((rgb >>> 8) & 0xff) / 255.0f;
				float blue = (rgb & 0xff) / 255.0f;
				int pixel = y * width + x;
				if (channels == 1) {
					matrix.values[pixel][0] = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
				} else {
					int plane = width * height;
					matrix.values[pixel][0] = red;
					matrix.values[plane + pixel][0] = green;
					matrix.values[2 * plane + pixel][0] = blue;
				}
			}
		}
		return matrix;
	}

	/** Converts a flattened one-channel or RGB matrix to a BufferedImage. */
	public static BufferedImage toBufferedImage(Matrix matrix, int width, int height, int channels) {
		if (matrix == null)
			throw new IllegalArgumentException("Matrix cannot be null");
		if (width <= 0 || height <= 0)
			throw new IllegalArgumentException("Image dimensions must be positive");
		validateChannels(channels);
		long expectedRows = (long) width * height * channels;
		if (expectedRows > Integer.MAX_VALUE || matrix.rows() != (int) expectedRows || matrix.columns() != 1)
			throw new IllegalArgumentException("Image matrix dimensions do not match width, height, and channels");

		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int plane = width * height;
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int pixel = y * width + x;
				int red;
				int green;
				int blue;
				if (channels == 1) {
					int gray = toByte(matrix.values[pixel][0]);
					red = green = blue = gray;
				} else {
					red = toByte(matrix.values[pixel][0]);
					green = toByte(matrix.values[plane + pixel][0]);
					blue = toByte(matrix.values[2 * plane + pixel][0]);
				}
				image.setRGB(x, y, (red << 16) | (green << 8) | blue);
			}
		}
		return image;
	}

	/** Saves a flattened one-channel or RGB matrix as an image file. */
	public static void save(Matrix matrix, int width, int height, int channels, String filename) throws IOException {
		save(matrix, width, height, channels, Path.of(filename));
	}

	/** Saves a flattened one-channel or RGB matrix as an image file. */
	public static void save(Matrix matrix, int width, int height, int channels, Path path) throws IOException {
		BufferedImage image = toBufferedImage(matrix, width, height, channels);
		String format = format(path);
		if (!ImageIO.write(image, format, path.toFile()))
			throw new IOException("Unsupported image format: " + format);
	}

	private static BufferedImage resize(BufferedImage image, int width, int height) {
		BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics2D graphics = resized.createGraphics();
		try {
			graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			graphics.drawImage(image, 0, 0, width, height, null);
		} finally {
			graphics.dispose();
		}
		return resized;
	}

	private static int toByte(float value) {
		float clamped = Math.max(0.0f, Math.min(1.0f, value));
		return Math.round(clamped * 255.0f);
	}

	private static void validateChannels(int channels) {
		if (channels != 1 && channels != 3)
			throw new IllegalArgumentException("Image channels must be 1 or 3");
	}

	private static String format(Path path) {
		String name = path.getFileName().toString();
		int dot = name.lastIndexOf('.');
		if (dot < 0 || dot == name.length() - 1)
			return "png";
		return name.substring(dot + 1).toLowerCase();
	}
}
