package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.transformer.TransformerMath;

/** Integer token IDs [1,tokens] to embeddings [hidden,tokens]. */
public final class EmbeddingLayer extends TrainableLayer {
    private final Matrix weights;
    private int[] tokens;
    private Matrix ids;

    public EmbeddingLayer(int vocabulary, int hidden) {
        if (vocabulary < 1 || hidden < 1) throw new IllegalArgumentException();
        weights = parameter(TransformerMath.random(hidden, vocabulary));
    }

    public Matrix getWeights() {
        return weights;
    }

    @Override
    public Matrix forward(Matrix input) {
        TransformerMath.host(input);
        if (TransformerMath.gpu() == null) TransformerMath.host(weights);
        if (input.rows() != 1 || input.columns() < 1)
            throw new IllegalArgumentException("Expected token IDs [1,tokens]");
        ids = input;
        tokens = new int[input.columns()];
        if (TransformerMath.gpu() != null) {
            for (int t = 0; t < tokens.length; t++) {
                float v = input.values[0][t];
                int id = (int) v;
                if (v != id || id < 0 || id >= weights.columns())
                    throw new IllegalArgumentException("Token outside vocabulary");
                tokens[t] = id;
            }
            return TransformerMath.gpu().embedding(input, weights);
        }
        Matrix out = new Matrix(weights.rows(), tokens.length);
        for (int t = 0; t < tokens.length; t++) {
            float v = input.values[0][t];
            int id = (int) v;
            if (v != id || id < 0 || id >= weights.columns())
                throw new IllegalArgumentException("Token outside vocabulary");
            tokens[t] = id;
            for (int i = 0; i < weights.rows(); i++) out.values[i][t] = weights.values[i][id];
        }
        return out;
    }

    @Override
    public Matrix backwardGradient(Matrix g) {
        if (tokens == null) throw new IllegalStateException("Forward required");
        if (g.rows() != weights.rows() || g.columns() != tokens.length)
            throw new IllegalArgumentException();
        if (TransformerMath.gpu() != null) {
            TransformerMath.gpu().embeddingBackward(ids, g, gradients.get(0));
            return new Matrix(1, tokens.length);
        }
        TransformerMath.host(g);
        Matrix dw = gradients.get(0);
        for (int t = 0; t < tokens.length; t++)
            for (int i = 0; i < weights.rows(); i++) dw.values[i][tokens[t]] += g.values[i][t];
        dw.markDirty();
        return new Matrix(1, tokens.length);
    }

    @Override
    public void clearTrainingState() {
        tokens = null;
        ids = null;
    }
}
