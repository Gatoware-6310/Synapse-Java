package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.Transformer;
import xyz.gatoware.synapse.transformer.*;

import java.io.*;

/** Versioned extension payload for reusable transformer layers in NeuralNetwork .snn v3. */
public final class TransformerLayerIO {
    private TransformerLayerIO() {}

    public static void write(DataOutput out, TrainableLayer layer) throws IOException {
        out.writeInt(1);
        if (layer instanceof Transformer t) {
            out.writeByte(8);
            TransformerIO.writeConfig(out, t.getConfig());
        } else if (layer instanceof EmbeddingLayer e) {
            out.writeByte(1);
            out.writeInt(e.getWeights().columns());
            out.writeInt(e.getWeights().rows());
        } else if (layer instanceof RMSNormLayer n) {
            out.writeByte(2);
            out.writeInt(n.scale.rows());
            out.writeFloat(n.epsilon);
            out.writeBoolean(n.center);
        } else if (layer instanceof FeedForwardLayer f) {
            out.writeByte(3);
            out.writeInt(f.up.columns());
            out.writeInt(f.up.rows());
            out.writeUTF(f.activation.name());
        } else if (layer instanceof MultiHeadAttentionLayer a) {
            out.writeByte(4);
            TransformerIO.writeConfig(
                    out,
                    new TransformerConfig(2, a.hidden, 1, a.heads)
                            .setKVHeads(a.kvHeads)
                            .setContextLength(a.context)
                            .setPositionEncoding(
                                    a.rope ? PositionEncoding.ROPE : PositionEncoding.LEARNED));
        } else if (layer instanceof MoELayer m) {
            out.writeByte(5);
            out.writeInt(m.router.columns());
            out.writeInt(m.children.get(0).parameters().get(0).rows());
            out.writeUTF(((FeedForwardLayer) m.children.get(0)).activation.name());
            TransformerIO.writeMoE(out, m.config);
        } else if (layer instanceof TransformerDecoderLayer d) {
            out.writeByte(6);
            TransformerIO.writeConfig(out, d.config);
            out.writeInt(d.index);
        } else throw new IOException("Unsupported trainable layer " + layer.getClass());
        TransformerIO.writeParameters(out, layer);
    }

    public static TrainableLayer read(DataInput in) throws IOException {
        if (in.readInt() != 1) throw new IOException("Unsupported transformer layer version");
        try {
            TrainableLayer layer =
                    switch (in.readUnsignedByte()) {
                        case 1 -> new EmbeddingLayer(in.readInt(), in.readInt());
                        case 2 -> new RMSNormLayer(in.readInt(), in.readFloat(), in.readBoolean());
                        case 3 ->
                                new FeedForwardLayer(
                                        in.readInt(),
                                        in.readInt(),
                                        Activations.valueOf(in.readUTF()));
                        case 4 -> new MultiHeadAttentionLayer(TransformerIO.readConfig(in));
                        case 5 ->
                                new MoELayer(
                                        in.readInt(),
                                        in.readInt(),
                                        Activations.valueOf(in.readUTF()),
                                        TransformerIO.readMoE(in));
                        case 6 ->
                                new TransformerDecoderLayer(
                                        TransformerIO.readConfig(in), in.readInt());
                        case 8 -> new Transformer(TransformerIO.readConfig(in));
                        default -> throw new IOException("Unknown transformer layer type");
                    };
            TransformerIO.readParameters(in, layer);
            return layer;
        } catch (IllegalArgumentException e) {
            throw new IOException("Invalid transformer layer", e);
        }
    }
}
