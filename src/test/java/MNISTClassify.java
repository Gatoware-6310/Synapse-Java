import java.awt.*;
import java.awt.event.*;
import java.io.IOException;
import javax.swing.*;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.NeuralNetwork;

public class MNISTClassify {

	private static final int GRID_SIZE = 28;
	private static final int SCALE = 14;

	public static void main(String[] args) throws IOException {
		NeuralNetwork network = NeuralNetwork.load("MNIST_example.snn");

		SwingUtilities.invokeLater(() -> {
			JFrame frame = new JFrame("Synapse MNIST Classifier");
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			frame.setLayout(new BorderLayout(10, 10));

			DrawingPanel drawingPanel = new DrawingPanel();

			JLabel classificationLabel = new JLabel(
					"Classification: ",
					SwingConstants.CENTER);

			classificationLabel.setFont(
					new Font(Font.SANS_SERIF, Font.BOLD, 20));

			JButton classifyButton = new JButton("Classify");
			JButton clearButton = new JButton("Clear");

			classifyButton.addActionListener(e -> {
				Matrix input = drawingPanel.toMatrix();

				int prediction = network.predict(input);

				classificationLabel.setText(
						"Classification: " + prediction);
			});

			clearButton.addActionListener(e -> {
				drawingPanel.clear();
				classificationLabel.setText("Classification: ");
			});

			JPanel buttons = new JPanel();
			buttons.add(clearButton);
			buttons.add(classifyButton);

			JPanel bottom = new JPanel(new BorderLayout());
			bottom.add(classificationLabel, BorderLayout.CENTER);
			bottom.add(buttons, BorderLayout.SOUTH);

			frame.add(drawingPanel, BorderLayout.CENTER);
			frame.add(bottom, BorderLayout.SOUTH);

			frame.pack();
			frame.setLocationRelativeTo(null);
			frame.setResizable(false);
			frame.setVisible(true);
		});
	}

	static class DrawingPanel extends JPanel {

		private final float[][] pixels = new float[GRID_SIZE][GRID_SIZE];

		DrawingPanel() {
			setPreferredSize(
					new Dimension(GRID_SIZE * SCALE, GRID_SIZE * SCALE));

			setBackground(Color.BLACK);

			MouseAdapter mouse = new MouseAdapter() {
				@Override
				public void mousePressed(MouseEvent e) {
					draw(e);
				}

				@Override
				public void mouseDragged(MouseEvent e) {
					draw(e);
				}
			};

			addMouseListener(mouse);
			addMouseMotionListener(mouse);
		}

		private void draw(MouseEvent e) {
			int x = e.getX() / SCALE;
			int y = e.getY() / SCALE;

			if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) {
				return;
			}

			// Slightly thicker brush than a single MNIST pixel
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int px = x + dx;
					int py = y + dy;

					if (px >= 0 && px < GRID_SIZE &&
							py >= 0 && py < GRID_SIZE) {
						float distance = Math.abs(dx) + Math.abs(dy);

						if (distance == 0) {
							pixels[py][px] = 1.0f;
						} else {
							pixels[py][px] = Math.max(
									pixels[py][px],
									0.5f);
						}
					}
				}
			}

			repaint();
		}

		public Matrix toMatrix() {
			float[][] flattened = new float[GRID_SIZE * GRID_SIZE][1];

			int index = 0;

			for (int y = 0; y < GRID_SIZE; y++) {
				for (int x = 0; x < GRID_SIZE; x++) {
					flattened[index++][0] = pixels[y][x];
				}
			}

			return new Matrix(flattened);
		}

		public void clear() {
			for (int y = 0; y < GRID_SIZE; y++) {
				for (int x = 0; x < GRID_SIZE; x++) {
					pixels[y][x] = 0.0f;
				}
			}

			repaint();
		}

		@Override
		protected void paintComponent(Graphics g) {
			super.paintComponent(g);

			for (int y = 0; y < GRID_SIZE; y++) {
				for (int x = 0; x < GRID_SIZE; x++) {
					float value = pixels[y][x];

					int brightness = (int) (value * 255);

					g.setColor(
							new Color(
									brightness,
									brightness,
									brightness));

					g.fillRect(
							x * SCALE,
							y * SCALE,
							SCALE,
							SCALE);
				}
			}
		}
	}
}
