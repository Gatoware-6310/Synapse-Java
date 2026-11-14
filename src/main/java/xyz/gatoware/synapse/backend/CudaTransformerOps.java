package xyz.gatoware.synapse.backend;

import jcuda.Pointer;
import jcuda.driver.*;
import jcuda.nvrtc.*;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Resident FP32 transformer operations sharing the existing backend's cuBLAS handle and memory
 * pool. Kernel failures are explicit: selecting CUDA never silently executes these operations on
 * CPU.
 */
public final class CudaTransformerOps implements AutoCloseable {
    private final CudaBackend cuda;
    private final CUmodule module = new CUmodule();
    private final Map<String, CUfunction> functions = new HashMap<>();

    CudaTransformerOps(CudaBackend cuda) {
        this.cuda = cuda;
        nvrtcProgram program = new nvrtcProgram();
        boolean created = false;
        try {
            String source;
            try (var stream = CudaTransformerOps.class.getResourceAsStream("transformer.cu")) {
                if (stream == null) throw new IOException("Missing transformer CUDA resource");
                source = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            }
            JNvrtc.setExceptionsEnabled(true);
            JCudaDriver.setExceptionsEnabled(true);
            JNvrtc.nvrtcCreateProgram(program, source, "transformer.cu", 0, null, null);
            created = true;
            int[] device = {0};
            jcuda.runtime.JCuda.cudaGetDevice(device);
            jcuda.runtime.cudaDeviceProp properties = new jcuda.runtime.cudaDeviceProp();
            jcuda.runtime.JCuda.cudaGetDeviceProperties(properties, device[0]);
            String[] options = {
                "--gpu-architecture=compute_" + properties.major + properties.minor
            };
            try {
                JNvrtc.nvrtcCompileProgram(program, options.length, options);
            } catch (RuntimeException ex) {
                String[] log = {""};
                JNvrtc.nvrtcGetProgramLog(program, log);
                throw new IllegalStateException(
                        "Transformer CUDA compilation failed: " + log[0], ex);
            }
            String[] ptx = new String[1];
            JNvrtc.nvrtcGetPTX(program, ptx);
            JCudaDriver.cuModuleLoadData(module, ptx[0]);
        } catch (IOException ex) {
            throw new IllegalStateException("Cannot load transformer kernels", ex);
        } finally {
            if (created) JNvrtc.nvrtcDestroyProgram(program);
        }
    }

    private Pointer ptr(Matrix m) {
        return Pointer.to(cuda.device(m).pointer);
    }

    private Pointer arg(int n) {
        return Pointer.to(new int[] {n});
    }

    private Pointer arg(float n) {
        return Pointer.to(new float[] {n});
    }

    private void launch(String name, int n, Pointer... args) {
        CUfunction f =
                functions.computeIfAbsent(
                        name,
                        k -> {
                            CUfunction fn = new CUfunction();
                            JCudaDriver.cuModuleGetFunction(fn, module, k);
                            return fn;
                        });
        JCudaDriver.cuLaunchKernel(
                f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, null, Pointer.to(args), null);
    }

    private Matrix allocate(int r, int c) {
        return cuda.resident(cuda.temporary(Math.multiplyExact(r, c)), r, c);
    }

    public Matrix zeros(int r, int c) {
        Matrix y = allocate(r, c);
        cuda.zero(cuda.device(y));
        return y;
    }

    public Matrix mm(Matrix a, Matrix b) {
        return cuda.transformerMultiply(a, b);
    }

    public Matrix transpose(Matrix x) {
        Matrix y = allocate(x.columns(), x.rows());
        launch(
                "transpose",
                x.rows() * x.columns(),
                ptr(x),
                ptr(y),
                arg(x.rows()),
                arg(x.columns()));
        return y;
    }

    public Matrix add(Matrix a, Matrix b) {
        return binary(a, b, 0, 1);
    }

    public Matrix scale(Matrix a, float scale) {
        return binary(a, a, 2, scale);
    }

    private Matrix binary(Matrix a, Matrix b, int kind, float scale) {
        Matrix y = allocate(a.rows(), a.columns());
        launch(
                "binary",
                a.rows() * a.columns(),
                ptr(a),
                ptr(b),
                ptr(y),
                arg(a.rows() * a.columns()),
                arg(kind),
                arg(scale));
        return y;
    }

    public void accumulate(Matrix a, Matrix b) {
        launch(
                "binary",
                a.rows() * a.columns(),
                ptr(a),
                ptr(b),
                ptr(a),
                arg(a.rows() * a.columns()),
                arg(0),
                arg(1f));
        cuda.device(a).deviceDirty = true;
    }

    public Matrix activation(Matrix u, Matrix v, int kind) {
        Matrix y = allocate(u.rows(), u.columns());
        launch(
                "activation",
                u.rows() * u.columns(),
                ptr(u),
                ptr(v == null ? u : v),
                ptr(u),
                ptr(y),
                ptr(y),
                arg(u.rows() * u.columns()),
                arg(kind),
                arg(0));
        return y;
    }

    public Matrix[] activationBackward(Matrix u, Matrix v, Matrix g, int kind) {
        Matrix du = allocate(u.rows(), u.columns()), dv = allocate(u.rows(), u.columns());
        launch(
                "activation",
                u.rows() * u.columns(),
                ptr(u),
                ptr(v == null ? u : v),
                ptr(g),
                ptr(du),
                ptr(dv),
                arg(u.rows() * u.columns()),
                arg(kind),
                arg(1));
        return new Matrix[] {du, dv};
    }

    public Matrix[] norm(Matrix x, Matrix w, Matrix b, float eps) {
        Matrix n = allocate(x.rows(), x.columns()),
                inv = allocate(1, x.columns()),
                y = allocate(x.rows(), x.columns());
        launch(
                "norm_forward",
                x.columns(),
                ptr(x),
                ptr(w),
                ptr(b == null ? w : b),
                ptr(n),
                ptr(inv),
                ptr(y),
                arg(x.rows()),
                arg(x.columns()),
                arg(eps),
                arg(b == null ? 0 : 1));
        return new Matrix[] {y, n, inv};
    }

    public Matrix normBackward(Matrix g, Matrix n, Matrix inv, Matrix w, Matrix dw, Matrix db) {
        Matrix dx = allocate(g.rows(), g.columns());
        launch(
                "norm_backward",
                g.columns(),
                ptr(g),
                ptr(n),
                ptr(inv),
                ptr(w),
                ptr(dx),
                arg(g.rows()),
                arg(g.columns()),
                arg(db == null ? 0 : 1));
        launch(
                "norm_parameters",
                g.rows(),
                ptr(g),
                ptr(n),
                ptr(dw),
                ptr(db == null ? dw : db),
                arg(g.rows()),
                arg(g.columns()),
                arg(db == null ? 0 : 1));
        cuda.device(dw).deviceDirty = true;
        if (db != null) cuda.device(db).deviceDirty = true;
        return dx;
    }

    public void rope(Matrix x, int heads, int dim, int offset, boolean inverse) {
        launch(
                "rope",
                x.rows() * x.columns() / 2,
                ptr(x),
                arg(heads),
                arg(dim),
                arg(x.columns()),
                arg(offset),
                arg(inverse ? 1 : 0));
        cuda.device(x).deviceDirty = true;
    }

    public Matrix attention(Matrix q, Matrix k, Matrix v, int heads, int kv, int dim, int offset) {
        Matrix y = allocate(q.rows(), q.columns());
        CUfunction f =
                functions.computeIfAbsent(
                        "attention",
                        name -> {
                            CUfunction fn = new CUfunction();
                            JCudaDriver.cuModuleGetFunction(fn, module, name);
                            return fn;
                        });
        JCudaDriver.cuLaunchKernel(
                f,
                heads * q.columns(),
                1,
                1,
                256,
                1,
                1,
                (256 + dim) * 4,
                null,
                Pointer.to(
                        ptr(q),
                        ptr(k),
                        ptr(v),
                        ptr(y),
                        arg(heads),
                        arg(kv),
                        arg(dim),
                        arg(q.columns()),
                        arg(k.columns()),
                        arg(offset),
                        arg(0)),
                null);
        return y;
    }

    private final java.util.Set<KVCache> kvCaches =
            java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());

    public final class KVCache implements AutoCloseable {
        private final CudaBackend.DeviceBuffer keys, values;
        private final int rows, context;
        private final boolean half;
        private boolean closed;

        private KVCache(int rows, int context, boolean half) {
            this.rows = rows;
            this.context = context;
            this.half = half;
            int count = Math.multiplyExact(rows, context),
                    units = half ? (int) (((long) count + 1) / 2) : count;
            keys = cuda.temporary(units);
            try {
                values = cuda.temporary(units);
            } catch (RuntimeException ex) {
                cuda.releaseTemporary(keys);
                throw ex;
            }
            kvCaches.add(this);
        }

        public void append(Matrix k, Matrix v, int offset) {
            if (closed) throw new IllegalStateException("Closed KV cache");
            launch(
                    "cache_append",
                    k.rows() * k.columns(),
                    ptr(k),
                    Pointer.to(keys.pointer),
                    arg(rows),
                    arg(k.columns()),
                    arg(context),
                    arg(offset),
                    arg(half ? 1 : 0));
            launch(
                    "cache_append",
                    v.rows() * v.columns(),
                    ptr(v),
                    Pointer.to(values.pointer),
                    arg(rows),
                    arg(v.columns()),
                    arg(context),
                    arg(offset),
                    arg(half ? 1 : 0));
        }

        @Override
        public void close() {
            if (!closed) {
                cuda.releaseTemporary(keys);
                cuda.releaseTemporary(values);
                closed = true;
                kvCaches.remove(this);
            }
        }
    }

    public KVCache newKVCache(int rows, int context, boolean half) {
        return new KVCache(rows, context, half);
    }

    public Matrix attentionCached(Matrix q, KVCache cache, int heads, int kv, int dim, int offset) {
        Matrix y = allocate(q.rows(), q.columns());
        CUfunction f =
                functions.computeIfAbsent(
                        "attention",
                        name -> {
                            CUfunction fn = new CUfunction();
                            JCudaDriver.cuModuleGetFunction(fn, module, name);
                            return fn;
                        });
        JCudaDriver.cuLaunchKernel(
                f,
                heads * q.columns(),
                1,
                1,
                256,
                1,
                1,
                (256 + dim) * 4,
                null,
                Pointer.to(
                        ptr(q),
                        Pointer.to(cache.keys.pointer),
                        Pointer.to(cache.values.pointer),
                        ptr(y),
                        arg(heads),
                        arg(kv),
                        arg(dim),
                        arg(q.columns()),
                        arg(cache.context),
                        arg(offset),
                        arg(cache.half ? 1 : 0)),
                null);
        return y;
    }

    public Matrix[] attentionBackward(
            Matrix q, Matrix k, Matrix v, Matrix y, Matrix g, int heads, int kv, int dim) {
        Matrix dq = zeros(q.rows(), q.columns()),
                dk = zeros(k.rows(), k.columns()),
                dv = zeros(v.rows(), v.columns());
        launch(
                "attention_backward",
                heads * q.columns(),
                ptr(q),
                ptr(k),
                ptr(v),
                ptr(y),
                ptr(g),
                ptr(dq),
                ptr(dk),
                ptr(dv),
                arg(heads),
                arg(kv),
                arg(dim),
                arg(q.columns()));
        return new Matrix[] {dq, dk, dv};
    }

    public Matrix embedding(Matrix ids, Matrix weights) {
        Matrix y = allocate(weights.rows(), ids.columns());
        launch(
                "embedding",
                y.rows() * y.columns(),
                ptr(ids),
                ptr(weights),
                ptr(y),
                arg(weights.rows()),
                arg(weights.columns()),
                arg(ids.columns()));
        return y;
    }

    public void embeddingBackward(Matrix ids, Matrix g, Matrix dw) {
        launch(
                "embedding_backward",
                g.rows() * g.columns(),
                ptr(ids),
                ptr(g),
                ptr(dw),
                arg(dw.rows()),
                arg(dw.columns()),
                arg(ids.columns()));
        cuda.device(dw).deviceDirty = true;
    }

    public Matrix slice(Matrix x, int offset, int width) {
        Matrix y = allocate(x.rows(), width);
        launch(
                "slice",
                y.rows() * width,
                ptr(x),
                ptr(y),
                arg(x.rows()),
                arg(x.columns()),
                arg(offset),
                arg(width));
        return y;
    }

    public void append(Matrix x, Matrix cache, int offset, boolean add) {
        launch(
                "append",
                x.rows() * x.columns(),
                ptr(x),
                ptr(cache),
                arg(x.rows()),
                arg(cache.columns()),
                arg(offset),
                arg(x.columns()),
                arg(add ? 1 : 0));
        cuda.device(cache).deviceDirty = true;
    }

    public record LossGradient(float loss, Matrix gradient) {}

    public LossGradient crossEntropy(Matrix logits, int[] sequence) {
        int n = sequence.length - 1;
        Matrix labels = new Matrix(1, n);
        for (int i = 0; i < n; i++) labels.values[0][i] = sequence[i + 1];
        Matrix delta = allocate(logits.rows(), n), losses = allocate(1, n);
        launch(
                "xent",
                n,
                ptr(logits),
                ptr(labels),
                ptr(delta),
                ptr(losses),
                arg(logits.rows()),
                arg(n));
        cuda.materialize(losses);
        float sum = 0;
        for (float v : losses.values[0]) sum += v;
        return new LossGradient(sum, delta);
    }

    public double squaredSum(Matrix g) {
        Matrix out = zeros(1, 1);
        launch(
                "squared_sum",
                Math.min(65536, g.rows() * g.columns()),
                ptr(g),
                ptr(out),
                arg(g.rows() * g.columns()));
        cuda.materialize(out);
        return out.values[0][0];
    }

    public double squaredSums(java.util.List<Matrix> gradients) {
        Matrix sums = zeros(1, gradients.size());
        for (int i = 0; i < gradients.size(); i++) {
            Matrix g = gradients.get(i);
            launch(
                    "squared_sum",
                    Math.min(65536, g.rows() * g.columns()),
                    ptr(g),
                    Pointer.to(cuda.device(sums).pointer.withByteOffset((long) i * 4)),
                    arg(g.rows() * g.columns()));
        }
        cuda.materialize(sums);
        double sum = 0;
        for (float value : sums.values[0]) sum += value;
        return sum;
    }

    public int greedy(Matrix logits, int token) {
        Matrix output = allocate(1, 1);
        launch(
                "greedy",
                256,
                ptr(logits),
                ptr(output),
                arg(logits.rows()),
                arg(logits.columns()),
                arg(token));
        cuda.materialize(output);
        return (int) output.values[0][0];
    }

    public void zero(Matrix g) {
        cuda.zero(cuda.device(g));
        cuda.device(g).deviceDirty = true;
    }

    public void update(Matrix p, Matrix g, Optimizer optimizer, float rate) {
        if (optimizer instanceof AdamW a) {
            Matrix decayed = scale(p, 1 - rate * a.getWeightDecay());
            launch(
                    "binary",
                    p.rows() * p.columns(),
                    ptr(decayed),
                    ptr(decayed),
                    ptr(p),
                    arg(p.rows() * p.columns()),
                    arg(2),
                    arg(1f));
            cuda.device(p).deviceDirty = true;
            cuda.updateResident(p, cuda.device(g), a.getAdam(), rate);
        } else cuda.updateResident(p, cuda.device(g), optimizer, rate);
    }

    public record Routing(
            Matrix probabilities,
            Matrix scores,
            Matrix weights,
            Matrix frequency,
            Matrix slots,
            int[] counts,
            int capacity,
            float loss) {}

    public Routing route(Matrix logits, int topK, int capacity, boolean sigmoid, float balance) {
        int e = logits.rows(), n = logits.columns();
        Matrix p = allocate(e, n),
                score = allocate(e, n),
                w = allocate(e, n),
                freq = zeros(e, 1),
                slots = allocate(e, capacity),
                counts = allocate(1, e);
        launch(
                "route",
                n,
                ptr(logits),
                ptr(p),
                ptr(score),
                ptr(w),
                ptr(freq),
                arg(e),
                arg(n),
                arg(topK),
                arg(sigmoid ? 1 : 0));
        launch("dispatch", e, ptr(w), ptr(slots), ptr(counts), arg(e), arg(n), arg(capacity));
        cuda.materialize(counts);
        int[] c = new int[e];
        for (int i = 0; i < e; i++) c[i] = (int) counts.values[0][i];
        float loss = 0;
        if (balance != 0) {
            Matrix losses = allocate(1, n);
            launch("balance_loss", n, ptr(p), ptr(freq), ptr(losses), arg(e), arg(n), arg(balance));
            cuda.materialize(losses);
            for (float v : losses.values[0]) loss += v;
        }
        return new Routing(p, score, w, freq, slots, c, capacity, loss);
    }

    public Matrix gather(Matrix x, Routing r, int expert) {
        int count = r.counts()[expert];
        Matrix y = allocate(x.rows(), count);
        launch(
                "gather",
                x.rows() * count,
                ptr(x),
                ptr(r.slots()),
                ptr(y),
                arg(x.rows()),
                arg(x.columns()),
                arg(r.capacity()),
                arg(expert),
                arg(count));
        return y;
    }

    public void combine(Matrix y, Routing r, int expert, Matrix out) {
        launch(
                "combine",
                y.rows() * y.columns(),
                ptr(y),
                ptr(r.weights()),
                ptr(r.slots()),
                ptr(out),
                arg(out.rows()),
                arg(out.columns()),
                arg(r.capacity()),
                arg(expert),
                arg(y.columns()));
        cuda.device(out).deviceDirty = true;
    }

    public Matrix expertDelta(Matrix g, Matrix y, Routing r, int expert, Matrix dw) {
        Matrix dy = allocate(y.rows(), y.columns());
        launch(
                "expert_delta",
                y.columns(),
                ptr(g),
                ptr(y),
                ptr(r.weights()),
                ptr(r.slots()),
                ptr(dy),
                ptr(dw),
                arg(g.rows()),
                arg(g.columns()),
                arg(r.capacity()),
                arg(expert),
                arg(y.columns()));
        cuda.device(dw).deviceDirty = true;
        return dy;
    }

    public void scatterGradient(Matrix batch, Routing r, int expert, Matrix dx) {
        launch(
                "scatter_gradient",
                batch.rows() * batch.columns(),
                ptr(batch),
                ptr(r.slots()),
                ptr(dx),
                arg(dx.rows()),
                arg(dx.columns()),
                arg(r.capacity()),
                arg(expert),
                arg(batch.columns()));
        cuda.device(dx).deviceDirty = true;
    }

    public Matrix routeBackward(Matrix dw, Routing r, boolean sigmoid, float balance) {
        Matrix dl = allocate(dw.rows(), dw.columns());
        launch(
                "route_backward",
                dw.columns(),
                ptr(dw),
                ptr(r.weights()),
                ptr(r.probabilities()),
                ptr(r.scores()),
                ptr(r.frequency()),
                ptr(dl),
                arg(dw.rows()),
                arg(dw.columns()),
                arg(sigmoid ? 1 : 0),
                arg(balance));
        return dl;
    }

    @Override
    public void close() {
        for (KVCache cache : new java.util.ArrayList<>(kvCaches)) cache.close();
        JCudaDriver.cuModuleUnload(module);
    }
}
