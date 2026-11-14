package xyz.gatoware.synapse.layer;

import java.util.*;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.optimizer.*;
import xyz.gatoware.synapse.transformer.TransformerMath;

/** Shared parameter/gradient lifecycle for composable transformer layers. Gradients are sums. */
public abstract class TrainableLayer implements Layer {
    protected final List<Matrix> parameters=new ArrayList<>(), gradients=new ArrayList<>();
    protected final List<TrainableLayer> children=new ArrayList<>();
    protected Matrix parameter(Matrix p) { parameters.add(p);gradients.add(new Matrix(p.rows(),p.columns()));return p; }
    public List<Matrix> parameters() { List<Matrix> p=new ArrayList<>(parameters);for(var c:children)p.addAll(c.parameters());return Collections.unmodifiableList(p); }
    public List<Matrix> gradients() { List<Matrix> p=new ArrayList<>(gradients);for(var c:children)p.addAll(c.gradients());return Collections.unmodifiableList(p); }
    public void zeroGrad() { for(Matrix g:gradients){TransformerMath.host(g); for(float[] r:g.values)Arrays.fill(r,0);g.markDirty();}for(var c:children)c.zeroGrad(); }
    /** Accumulates parameter gradients without applying an optimizer update. */
    public abstract Matrix backwardGradient(Matrix gradient);
    @Override public Matrix backwardInput(Matrix g) {
        List<Matrix> gs=gradients();List<Matrix> saved=new ArrayList<>();for(Matrix x:gs){TransformerMath.host(x);saved.add(x.copy());}
        Matrix result=backwardGradient(g);
        for(int i=0;i<gs.size();i++){Matrix a=gs.get(i),b=saved.get(i);for(int r=0;r<a.rows();r++)System.arraycopy(b.values[r],0,a.values[r],0,a.columns());a.markDirty();}
        return result;
    }
    @Override public Matrix backward(Matrix g,float rate) { return backward(g,rate,new SGD()); }
    @Override public Matrix backward(Matrix g,float rate,Optimizer optimizer) { if(optimizer==null||!Float.isFinite(rate)||rate<=0)throw new IllegalArgumentException("Invalid optimizer/rate");zeroGrad();Matrix result=backwardGradient(g);step(optimizer,rate);return result; }
    public void step(Optimizer optimizer,float rate) { List<Matrix> ps=parameters(),gs=gradients(); for(int i=0;i<ps.size();i++){TransformerMath.host(ps.get(i));TransformerMath.host(gs.get(i));optimizer.update(ps.get(i),gs.get(i),rate);ps.get(i).markDirty();} }
}
