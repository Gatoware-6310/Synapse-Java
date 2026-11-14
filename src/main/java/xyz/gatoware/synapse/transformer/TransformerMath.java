package xyz.gatoware.synapse.transformer;

import xyz.gatoware.synapse.Synapse;
import xyz.gatoware.synapse.backend.CudaBackend;
import xyz.gatoware.synapse.matrix.Matrix;

/** Matrix utilities shared by transformer layers; products use the selected Synapse backend. */
public final class TransformerMath {
    private TransformerMath() {}
    public static void host(Matrix m) { if(Synapse.backend() instanceof CudaBackend c) c.materialize(m); }
    public static Matrix transpose(Matrix x) { host(x); Matrix y=new Matrix(x.columns(),x.rows()); for(int i=0;i<x.rows();i++) for(int j=0;j<x.columns();j++) y.values[j][i]=x.values[i][j]; return y; }
    public static Matrix mm(Matrix a,Matrix b) { if(a.columns()!=b.rows()) throw new IllegalArgumentException("Incompatible product"); return new Matrix(Synapse.backend().multiply(a.values,b.values,a.rows(),a.columns(),b.columns())); }
    public static Matrix add(Matrix a,Matrix b) { host(a);host(b); if(a.rows()!=b.rows()||a.columns()!=b.columns()) throw new IllegalArgumentException("Incompatible sum"); Matrix y=new Matrix(a.rows(),a.columns());for(int i=0;i<a.rows();i++) for(int j=0;j<a.columns();j++)y.values[i][j]=a.values[i][j]+b.values[i][j]; return y; }
    public static Matrix random(int r,int c) { Matrix m=new Matrix(r,c); double s=1/Math.sqrt(c); for(float[] row:m.values) for(int j=0;j<c;j++) row[j]=(float)((Math.random()*2-1)*s); return m; }
    public static void accumulate(Matrix a,Matrix b) { host(a);host(b); for(int i=0;i<a.rows();i++)for(int j=0;j<a.columns();j++)a.values[i][j]+=b.values[i][j];a.markDirty(); }
    public static void shape(Matrix g,Matrix y) { if(y==null)throw new IllegalStateException("Forward must precede backward");if(g.rows()!=y.rows()||g.columns()!=y.columns())throw new IllegalArgumentException("Gradient shape mismatch");host(g); }
}
