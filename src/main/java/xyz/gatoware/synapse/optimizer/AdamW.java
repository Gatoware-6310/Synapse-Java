package xyz.gatoware.synapse.optimizer;

import xyz.gatoware.synapse.matrix.Matrix;

/** Adam with decoupled weight decay. State is keyed by parameter identity, as in Adam. */
public final class AdamW implements Optimizer {
    private final Adam adam;
    private final float weightDecay;

    public AdamW() {
        this(.01f);
    }

    public AdamW(float weightDecay) {
        if (!Float.isFinite(weightDecay) || weightDecay < 0) throw new IllegalArgumentException();
        this.weightDecay = weightDecay;
        adam = new Adam();
    }

    public Adam getAdam() {
        return adam;
    }

    public float getWeightDecay() {
        return weightDecay;
    }

    @Override
    public void update(Matrix p, Matrix g, float rate) {
        if (p.rows() != g.rows()
                || p.columns() != g.columns()
                || !Float.isFinite(rate)
                || rate <= 0) throw new IllegalArgumentException();
        for (float[] row : p.values)
            for (int j = 0; j < row.length; j++) row[j] *= 1 - rate * weightDecay;
        adam.update(p, g, rate);
        p.markDirty();
    }

    @Override
    public void reset() {
        adam.reset();
    }
}
