import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import xyz.gatoware.synapse.layer.*;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.transformer.*;

class TransformerLayerTest {
    static Matrix input(int rows,int cols){Matrix x=new Matrix(rows,cols);for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)x.values[i][j]=(float)Math.sin(i*1.3+j*.7);return x;}
    static double objective(TrainableLayer l,Matrix x,Matrix g){Matrix y=l.forward(x);double s=0;for(int i=0;i<y.rows();i++)for(int j=0;j<y.columns();j++)s+=y.values[i][j]*g.values[i][j];return s+(l instanceof MoELayer m?m.auxiliaryLoss():0);}
    static void check(TrainableLayer l,Matrix x){Matrix y=l.forward(x),g=input(y.rows(),y.columns());l.zeroGrad();Matrix dx=l.backwardGradient(g);var ps=l.parameters();var gs=l.gradients();float eps=.002f;
        for(int p=0;p<ps.size();p++){Matrix m=ps.get(p),d=gs.get(p).copy();for(int i=0;i<m.rows();i++)for(int j=0;j<m.columns();j++){float old=m.values[i][j];m.values[i][j]=old+eps;m.markDirty();double a=objective(l,x,g);m.values[i][j]=old-eps;m.markDirty();double b=objective(l,x,g);m.values[i][j]=old;m.markDirty();assertEquals((a-b)/(2*eps),d.values[i][j],.004,"parameter "+p+" ["+i+","+j+"]");}}
        if(!(l instanceof EmbeddingLayer)&&!(l instanceof xyz.gatoware.synapse.Transformer))for(int i=0;i<x.rows();i++)for(int j=0;j<x.columns();j++){float old=x.values[i][j];x.values[i][j]=old+eps;x.markDirty();double a=objective(l,x,g);x.values[i][j]=old-eps;x.markDirty();double b=objective(l,x,g);x.values[i][j]=old;x.markDirty();assertEquals((a-b)/(2*eps),dx.values[i][j],.004,"input gradient");}
    }
    @Test void normsAndFFNGradients(){check(new RMSNormLayer(4),input(4,3));check(new LayerNormLayer(4),input(4,3));for(var a:Activations.values())check(new FeedForwardLayer(4,5,a),input(4,3));}
    @Test void embeddingGradients(){check(new EmbeddingLayer(4,3),new Matrix(new float[][]{{1,2,1}}));}
    @Test void attentionGradientsAndCausality(){for(int kv:new int[]{1,2,4})for(var pos:PositionEncoding.values()){
        var c=new TransformerConfig(8,8,1,4).setKVHeads(kv).setPositionEncoding(pos);var l=new MultiHeadAttentionLayer(c);check(l,input(8,3));
        Matrix x=input(8,4),a=l.forward(x).copy();for(int i=0;i<8;i++)x.values[i][3]+=50;Matrix b=l.forward(x);for(int i=0;i<8;i++)for(int t=0;t<3;t++)assertEquals(a.values[i][t],b.values[i][t],1e-6);
        l.resetCache();for(int t=0;t<4;t++){Matrix token=new Matrix(8,1);for(int i=0;i<8;i++)token.values[i][0]=x.values[i][t];Matrix y=l.forwardCached(token);for(int i=0;i<8;i++)assertEquals(b.values[i][t],y.values[i][0],1e-5);}
    }}
    @Test void moeGradients(){for(var r:RouterType.values())check(new MoELayer(3,4,Activations.SWIGLU,new MoEConfig(3,2).setSharedExperts(1).setRouter(r).setLoadBalancing(.03f)),input(3,3));}
    @Test void decoderGradients(){for(boolean pre:new boolean[]{true,false})for(var norm:Normalization.values())check(new TransformerDecoderLayer(new TransformerConfig(8,4,1,2).setNormalization(norm).setPreNormalization(pre)),input(4,3));}
    @Test void validation(){assertThrows(IllegalArgumentException.class,()->new TransformerConfig(8,6,2,4));assertThrows(IllegalArgumentException.class,()->new TransformerConfig(8,8,2,4).setKVHeads(3));assertThrows(IllegalArgumentException.class,()->new MoEConfig(2,3));}
}
