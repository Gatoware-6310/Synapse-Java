package xyz.gatoware.synapse.layer;

import java.util.*;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.transformer.*;
import static xyz.gatoware.synapse.transformer.TransformerMath.*;

/** Token-choice top-k MoE with grouped expert batches, shared experts and differentiable balance loss. */
public final class MoELayer extends TrainableLayer {
    private final MoEConfig config;
    private final Matrix router;
    private final FeedForwardLayer[] experts,shared;
    private Matrix input,output,probability,score;
    private int[][] selected,dispatch;
    private Matrix[] expertOutput;
    private float[][] weights;
    private float[] frequency;
    private float auxiliaryLoss;
    public MoELayer(int hidden,int intermediate,Activations activation,MoEConfig config){
        this.config=config.copy();router=parameter(random(config.experts,hidden));experts=new FeedForwardLayer[config.experts];shared=new FeedForwardLayer[config.sharedExperts];int f=config.expertHiddenSize>0?config.expertHiddenSize:intermediate;
        for(int i=0;i<experts.length;i++){experts[i]=new FeedForwardLayer(hidden,f,activation);children.add(experts[i]);}for(int i=0;i<shared.length;i++){shared[i]=new FeedForwardLayer(hidden,f,activation);children.add(shared[i]);}
    }
    public float auxiliaryLoss(){return auxiliaryLoss;}
    public int[][] selectedExperts(){if(selected==null)throw new IllegalStateException();int[][] a=new int[selected.length][];for(int i=0;i<a.length;i++)a[i]=selected[i].clone();return a;}
    @Override public Matrix forward(Matrix x){
        if(x.columns()<1)throw new IllegalArgumentException();input=x;host(x);Matrix logits=mm(router,x);int n=x.columns(),e=experts.length,k=config.topK;
        probability=new Matrix(e,n);score=new Matrix(e,n);selected=new int[n][k];weights=new float[n][k];frequency=new float[e];
        for(int t=0;t<n;t++){
            double max=-Double.MAX_VALUE;for(int j=0;j<e;j++)max=Math.max(max,logits.values[j][t]);double sum=0;for(int j=0;j<e;j++){float p=(float)Math.exp(logits.values[j][t]-max);probability.values[j][t]=p;sum+=p;}for(int j=0;j<e;j++){probability.values[j][t]/=sum;score.values[j][t]=config.router==RouterType.TOP_K_SOFTMAX?probability.values[j][t]:(float)(1/(1+Math.exp(-logits.values[j][t])));}
            boolean[] used=new boolean[e];double denom=0;for(int j=0;j<k;j++){int best=-1;for(int z=0;z<e;z++)if(!used[z]&&(best<0||score.values[z][t]>score.values[best][t]))best=z;selected[t][j]=best;used[best]=true;denom+=score.values[best][t];frequency[best]+=1f/(n*k);}for(int j=0;j<k;j++)weights[t][j]=(float)(denom>0?score.values[selected[t][j]][t]/denom:1.0/k);
        }
        auxiliaryLoss=0;for(int j=0;j<e;j++)for(int t=0;t<n;t++)auxiliaryLoss+=config.loadBalancing*e*frequency[j]*probability.values[j][t]/n;
        int capacity=Float.isInfinite(config.capacityFactor)?n*k:Math.max(1,(int)Math.ceil(config.capacityFactor*n*k/e));
        dispatch=new int[e][];expertOutput=new Matrix[e];output=new Matrix(x.rows(),n);
        for(int j=0;j<e;j++){
            int[] slots=new int[Math.min(n,capacity)];int count=0;for(int t=0;t<n;t++)for(int z=0;z<k;z++)if(selected[t][z]==j&&count<slots.length)slots[count++]=t*k+z;
            dispatch[j]=Arrays.copyOf(slots,count);if(count==0)continue;Matrix batch=new Matrix(x.rows(),count);for(int b=0;b<count;b++)for(int h=0;h<x.rows();h++)batch.values[h][b]=x.values[h][slots[b]/k];Matrix y=experts[j].forward(batch);expertOutput[j]=y;
            for(int b=0;b<count;b++){int t=slots[b]/k,z=slots[b]%k;for(int h=0;h<x.rows();h++)output.values[h][t]+=weights[t][z]*y.values[h][b];}
        }
        for(var s:shared)output=add(output,s.forward(x));return output;
    }
    @Override public Matrix backwardGradient(Matrix g){
        shape(g,output);int n=g.columns(),e=experts.length,k=config.topK;Matrix dx=new Matrix(g.rows(),n);float[][] dw=new float[n][k];
        for(int j=0;j<e;j++){int[] slots=dispatch[j];if(slots.length==0)continue;Matrix dy=new Matrix(g.rows(),slots.length);for(int b=0;b<slots.length;b++){int t=slots[b]/k,z=slots[b]%k;for(int h=0;h<g.rows();h++){dy.values[h][b]=g.values[h][t]*weights[t][z];dw[t][z]+=g.values[h][t]*expertOutput[j].values[h][b];}}Matrix part=experts[j].backwardGradient(dy);for(int b=0;b<slots.length;b++)for(int h=0;h<g.rows();h++)dx.values[h][slots[b]/k]+=part.values[h][b];}
        Matrix dl=new Matrix(e,n);
        for(int t=0;t<n;t++){
            double dot=0;for(int z=0;z<k;z++)dot+=weights[t][z]*dw[t][z];
            for(int z=0;z<k;z++){int j=selected[t][z];dl.values[j][t]+=(float)(weights[t][z]*(dw[t][z]-dot)*(config.router==RouterType.TOP_K_SIGMOID?1-score.values[j][t]:1));}
            double auxMean=0;for(int j=0;j<e;j++)auxMean+=probability.values[j][t]*frequency[j];for(int j=0;j<e;j++)dl.values[j][t]+=config.loadBalancing*e/n*probability.values[j][t]*(frequency[j]-auxMean);
        }
        accumulate(gradients.get(0),mm(dl,transpose(input)));dx=add(dx,mm(transpose(router),dl));for(var s:shared)dx=add(dx,s.backwardGradient(g));return dx;
    }
    @Override public void clearTrainingState(){input=null;output=null;probability=null;score=null;selected=null;dispatch=null;expertOutput=null;weights=null;frequency=null;super.clearTrainingState();}
}
