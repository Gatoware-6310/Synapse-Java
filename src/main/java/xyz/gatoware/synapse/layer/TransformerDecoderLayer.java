package xyz.gatoware.synapse.layer;

import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.transformer.*;
import static xyz.gatoware.synapse.transformer.TransformerMath.*;

/** Decoder block with causal attention, residual connections and pre/post normalization. */
public final class TransformerDecoderLayer extends TrainableLayer {
    private final MultiHeadAttentionLayer attention;
    private final RMSNormLayer norm1,norm2;
    private final TrainableLayer ffn;
    private final boolean pre;
    private Matrix output;
    public TransformerDecoderLayer(TransformerConfig c){this(c,0);}
    public TransformerDecoderLayer(TransformerConfig c,int index){
        pre=c.preNormalization;attention=new MultiHeadAttentionLayer(c);norm1=c.normalization==Normalization.RMS_NORM?new RMSNormLayer(c.hiddenSize):new LayerNormLayer(c.hiddenSize);norm2=c.normalization==Normalization.RMS_NORM?new RMSNormLayer(c.hiddenSize):new LayerNormLayer(c.hiddenSize);
        ffn=c.moe!=null&&c.moe.includes(index)?new MoELayer(c.hiddenSize,c.ffnSize,c.activation,c.moe):new FeedForwardLayer(c.hiddenSize,c.ffnSize,c.activation);
        children.add(attention);children.add(norm1);children.add(norm2);children.add(ffn);
    }
    public float auxiliaryLoss(){return ffn instanceof MoELayer m?m.auxiliaryLoss():0;}
    public void resetCache(){attention.resetCache();}
    @Override public Matrix forward(Matrix x){return run(x,false);}
    public Matrix forwardCached(Matrix x){return run(x,true);}
    private Matrix run(Matrix x,boolean cache){Matrix a=pre?norm1.forward(x):x;Matrix att=cache?attention.forwardCached(a):attention.forward(a);Matrix y=add(x,att);if(!pre)y=norm1.forward(y);Matrix z=add(y,ffn.forward(pre?norm2.forward(y):y));return output=pre?z:norm2.forward(z);}
    @Override public Matrix backwardGradient(Matrix g){shape(g,output);if(pre){Matrix dy=add(g,norm2.backwardGradient(ffn.backwardGradient(g)));return add(dy,norm1.backwardGradient(attention.backwardGradient(dy)));}Matrix dz=norm2.backwardGradient(g);Matrix dy=norm1.backwardGradient(add(dz,ffn.backwardGradient(dz)));return add(dy,attention.backwardGradient(dy));}
    @Override public void clearTrainingState(){output=null;super.clearTrainingState();}
}
