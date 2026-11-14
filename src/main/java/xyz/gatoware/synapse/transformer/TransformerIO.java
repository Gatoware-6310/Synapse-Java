package xyz.gatoware.synapse.transformer;

import java.io.*;
import xyz.gatoware.synapse.layer.TrainableLayer;
import xyz.gatoware.synapse.matrix.Matrix;

/** Explicit binary encoding; never uses Java object deserialization. */
public final class TransformerIO {
    private TransformerIO(){}
    public static void writeConfig(DataOutput out,TransformerConfig c)throws IOException{
        out.writeInt(c.vocabularySize);out.writeInt(c.hiddenSize);out.writeInt(c.layerCount);out.writeInt(c.heads);out.writeInt(c.contextLength);out.writeInt(c.kvHeads);out.writeInt(c.ffnSize);out.writeUTF(c.positionEncoding.name());out.writeUTF(c.normalization.name());out.writeUTF(c.activation.name());out.writeBoolean(c.preNormalization);out.writeBoolean(c.tieEmbeddings);out.writeBoolean(c.activationCheckpointing);out.writeBoolean(c.moe!=null);if(c.moe!=null)writeMoE(out,c.moe);
    }
    public static TransformerConfig readConfig(DataInput in)throws IOException{
        try{TransformerConfig c=new TransformerConfig(in.readInt(),in.readInt(),in.readInt(),in.readInt()).setContextLength(in.readInt()).setKVHeads(in.readInt()).setFFNSize(in.readInt()).setPositionEncoding(PositionEncoding.valueOf(in.readUTF())).setNormalization(Normalization.valueOf(in.readUTF())).setActivation(Activations.valueOf(in.readUTF())).setPreNormalization(in.readBoolean()).setTieEmbeddings(in.readBoolean()).setActivationCheckpointing(in.readBoolean());if(in.readBoolean())c.setMoE(readMoE(in));c.validate();return c;}catch(IllegalArgumentException e){throw new IOException("Invalid transformer configuration",e);}
    }
    public static void writeMoE(DataOutput out,MoEConfig c)throws IOException{out.writeInt(c.experts);out.writeInt(c.topK);out.writeInt(c.expertHiddenSize);out.writeInt(c.sharedExperts);out.writeInt(c.interval);out.writeFloat(c.loadBalancing);out.writeFloat(c.capacityFactor);out.writeUTF(c.router.name());out.writeInt(c.layers==null?-1:c.layers.length);if(c.layers!=null)for(int n:c.layers)out.writeInt(n);}
    public static MoEConfig readMoE(DataInput in)throws IOException{try{MoEConfig c=new MoEConfig(in.readInt(),in.readInt());int f=in.readInt();if(f!=0)c.setExpertHiddenSize(f);c.setSharedExperts(in.readInt()).setInterval(in.readInt()).setLoadBalancing(in.readFloat()).setCapacityFactor(in.readFloat()).setRouter(RouterType.valueOf(in.readUTF()));int n=in.readInt();if(n< -1||n>1_000_000)throw new IOException("Invalid MoE layer count");if(n>=0){int[] a=new int[n];for(int i=0;i<n;i++)a[i]=in.readInt();c.setLayers(a);}return c;}catch(IllegalArgumentException e){throw new IOException("Invalid MoE configuration",e);}}
    public static void writeParameters(DataOutput out,TrainableLayer layer)throws IOException{var ps=layer.parameters();out.writeInt(ps.size());for(Matrix p:ps){TransformerMath.host(p);out.writeInt(p.rows());out.writeInt(p.columns());for(float[] row:p.values)for(float v:row)out.writeFloat(v);}}
    public static void readParameters(DataInput in,TrainableLayer layer)throws IOException{var ps=layer.parameters();if(in.readInt()!=ps.size())throw new IOException("Parameter count mismatch");for(Matrix p:ps){if(in.readInt()!=p.rows()||in.readInt()!=p.columns())throw new IOException("Parameter shape mismatch");for(float[] row:p.values)for(int j=0;j<row.length;j++){float v=in.readFloat();if(!Float.isFinite(v))throw new IOException("Nonfinite parameter");row[j]=v;}p.markDirty();}}
}
