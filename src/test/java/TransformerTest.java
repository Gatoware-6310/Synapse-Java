import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import xyz.gatoware.synapse.*;
import xyz.gatoware.synapse.layer.*;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.*;
import xyz.gatoware.synapse.transformer.*;
import java.nio.file.*;
import java.util.*;

class TransformerTest {
    static TransformerConfig tiny(){return new TransformerConfig(4,8,2,4).setKVHeads(2).setContextLength(24).setFFNSize(12).setActivation(Activations.SWIGLU);}
    static void same(Matrix a,Matrix b,float tolerance){assertEquals(a.rows(),b.rows());assertEquals(a.columns(),b.columns());for(int i=0;i<a.rows();i++)for(int j=0;j<a.columns();j++)assertEquals(a.values[i][j],b.values[i][j],tolerance);}
    @Test void trainGenerateAndCheckpoint()throws Exception{for(boolean moe:new boolean[]{false,true}){
        var c=tiny();if(moe)c.setMoE(new MoEConfig(3,2).setSharedExperts(1).setLoadBalancing(.01f).setLayers(MoELayers.EVERY_OTHER));Transformer m=new Transformer(c);assertEquals(Transformer.estimatedParameterCount(c),m.parameterCount());int[][] data={{0,1,2,3,0,1,2,3,0},{1,2,3,0,1,2,3,0,1}};
        float before=m.loss(data[0]);m.fit(data,100,.01f,new AdamW(0),2,1);float after=m.loss(data[0]);System.out.println("MoE="+moe+" loss "+before+" -> "+after);assertTrue(after<before*.3,"loss should learn repeating tokens");
        int[] a=m.generate(new int[]{0,1},new GenerationConfig().setMaxNewTokens(8));int[] b=m.generate(new int[]{0,1},new GenerationConfig().setMaxNewTokens(8).setUseKVCache(false));assertArrayEquals(a,b);
        int[] sampled=m.generate(new int[]{0,1},new GenerationConfig().setMaxNewTokens(8).setTemperature(.8f).setTopK(3).setTopP(.9f).setSeed(42));assertArrayEquals(sampled,m.generate(new int[]{0,1},new GenerationConfig().setMaxNewTokens(8).setTemperature(.8f).setTopK(3).setTopP(.9f).setSeed(42)));
        Path p=Files.createTempFile("transformer",".snn");try{m.save(p);Transformer copy=Transformer.load(p);same(m.forward(data[0]),copy.forward(data[0]),0);assertArrayEquals(a,copy.generate(new int[]{0,1},new GenerationConfig().setMaxNewTokens(8)));}finally{Files.delete(p);}
    }}
    @Test void checkpointingMatchesAndLearnedPositions()throws Exception{for(boolean tied:new boolean[]{true,false})for(var pos:PositionEncoding.values()){
        var c=tiny().setTieEmbeddings(tied).setPositionEncoding(pos);var a=new Transformer(c);Path p=Files.createTempFile("transformer",".snn");try{a.save(p);var b=new Transformer(c.setActivationCheckpointing(true));var ap=a.parameters();var bp=b.parameters();for(int j=0;j<ap.size();j++)for(int i=0;i<ap.get(j).rows();i++)System.arraycopy(ap.get(j).values[i],0,bp.get(j).values[i],0,ap.get(j).columns());int[] tokens={1,2,3,1};same(a.forward(tokens),b.forward(tokens),0);Matrix g=TransformerLayerTest.input(4,4);a.zeroGrad();b.zeroGrad();a.backwardGradient(g);b.backwardGradient(g);for(int i=0;i<a.gradients().size();i++)same(a.gradients().get(i),b.gradients().get(i),1e-6f);
        b.resetCache();Matrix full=b.forward(tokens);for(int t=0;t<tokens.length;t++){Matrix y=b.forwardCached(new int[]{tokens[t]});for(int i=0;i<4;i++)assertEquals(full.values[i][t],y.values[i][0],1e-5);}
        }finally{Files.delete(p);}
    }}
    @Test void fullModelGradient(){for(boolean tied:new boolean[]{true,false}){Transformer m=new Transformer(new TransformerConfig(3,4,1,2).setFFNSize(5).setTieEmbeddings(tied).setPositionEncoding(PositionEncoding.LEARNED));TransformerLayerTest.check(m,new Matrix(new float[][]{{0,1,2}}));}}
    @Test void accumulationMatchesAverage(){Transformer a=new Transformer(tiny()),b=new Transformer(tiny());var ap=a.parameters();var bp=b.parameters();for(int j=0;j<ap.size();j++)for(int i=0;i<ap.get(j).rows();i++)System.arraycopy(ap.get(j).values[i],0,bp.get(j).values[i],0,ap.get(j).columns());int[] s={0,1,2,3};a.fit(new int[][]{s},1,.001f,new SGD(),1,0);b.fit(new int[][]{s,s},1,.001f,new SGD(),2,0);for(int j=0;j<ap.size();j++)same(ap.get(j),bp.get(j),1e-7f);}
}
