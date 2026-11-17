import xyz.gatoware.synapse.*;
import xyz.gatoware.synapse.layer.*;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.AdamW;
import xyz.gatoware.synapse.transformer.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Run: ./gradlew transformerExamples --args='train 30' (or basic/gqa/moe/layers/cuda). */
public final class TransformerExamples {
    static TransformerConfig config() {
        return new TransformerConfig(128, 32, 2, 4)
                .setContextLength(96)
                .setFFNSize(64)
                .setActivation(Activations.SWIGLU);
    }

    public static void main(String[] args) throws Exception {
        String example = args.length == 0 ? "basic" : args[0];
        if (example.equals("cuda")) {
            Synapse.useDevice(Devices.CUDA);
            example = "gqa";
        }
        if (example.equals("layers")) {
            NeuralNetwork network =
                    new NeuralNetwork(
                            new Layer[] {
                                new EmbeddingLayer(128, 32),
                                new TransformerDecoderLayer(config()),
                                new RMSNormLayer(32)
                            });
            Matrix result = network.forward(new Matrix(new float[][] {{65, 66, 67}}));
            TransformerMath.host(result);
            System.out.println("Standalone layers: " + result.rows() + " x " + result.columns());
            return;
        }
        var c = config();
        if (example.equals("gqa"))
            c.setKVHeads(1)
                    .setPositionEncoding(PositionEncoding.ROPE)
                    .setKVCachePrecision(KVCachePrecision.FP16);
        if (example.equals("moe"))
            c.setMoE(
                    new MoEConfig(4, 2)
                            .setSharedExperts(1)
                            .setLoadBalancing(.01f)
                            .setLayers(MoELayers.EVERY_OTHER));
        Transformer model = new Transformer(c);
        if (example.equals("train")) {
            String text = "hello synapse. hello java. hello synapse. hello java. ";
            int[] all = text.chars().toArray();
            List<int[]> sequences = new ArrayList<>();
            for (int start = 0; start + 17 <= all.length; start += 8)
                sequences.add(Arrays.copyOfRange(all, start, start + 17));
            int[][] data = sequences.toArray(int[][]::new);
            float before = model.loss(data[0]);
            model.fit(
                    data,
                    args.length > 1 ? Integer.parseInt(args[1]) : 30,
                    .005f,
                    new AdamW(),
                    2,
                    1);
            System.out.printf(
                    Locale.ROOT, "Training loss %.5f -> %.5f%n", before, model.loss(data[0]));
            Path checkpoint = Path.of("build/character-gpt.snn");
            Files.createDirectories(checkpoint.getParent());
            model.save(checkpoint);
            model = Transformer.load(checkpoint);
        }
        int[] output =
                model.generate(
                        "hello ".chars().toArray(),
                        new GenerationConfig()
                                .setMaxNewTokens(24)
                                .setTemperature(.8f)
                                .setTopK(12)
                                .setTopP(.95f));
        byte[] bytes = new byte[output.length];
        for (int i = 0; i < output.length; i++) bytes[i] = (byte) output[i];
        System.out.println(
                example.equals("train")
                        ? new String(bytes, StandardCharsets.US_ASCII)
                        : Arrays.toString(output));
        System.out.println(
                "Parameters: "
                        + model.parameterCount()
                        + "; conservative training payload estimate: "
                        + model.estimatedMemoryBytes(16, true)
                        + " bytes");
    }
}
