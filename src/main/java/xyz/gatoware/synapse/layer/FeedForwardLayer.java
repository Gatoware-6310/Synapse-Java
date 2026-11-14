package xyz.gatoware.synapse.layer;
import xyz.gatoware.synapse.matrix.Matrix;
import xyz.gatoware.synapse.transformer.*;
import static xyz.gatoware.synapse.transformer.TransformerMath.*;

/** Bias-free batched GELU or gated SwiGLU FFN. */
public final class FeedForwardLayer extends TrainableLayer {
    private final Matrix up,down,gate;private final Activations activation;
    private Matrix input,u,v,a,output;
    public FeedForwardLayer(int hidden,int intermediate,Activations activation){if(hidden<1||intermediate<1)throw new IllegalArgumentException();this.activation=java.util.Objects.requireNonNull(activation);up=parameter(random(intermediate,hidden));down=parameter(random(hidden,intermediate));gate=activation==Activations.SWIGLU?parameter(random(intermediate,hidden)):null;}
    public static float gelu(float x){return (float)(0.5*x*(1+Math.tanh(0.7978845608028654*(x+0.044715*x*x*x))));}
    public static float geluDerivative(float x){double t=Math.tanh(0.7978845608028654*(x+0.044715*x*x*x));return (float)(0.5*(1+t)+0.5*x*(1-t*t)*0.7978845608028654*(1+3*0.044715*x*x));}
    @Override public Matrix forward(Matrix x){input=x;u=mm(up,x);v=gate==null?null:mm(gate,x);a=new Matrix(u.rows(),u.columns());for(int i=0;i<u.rows();i++)for(int t=0;t<u.columns();t++){float z=u.values[i][t];a.values[i][t]=gate==null?gelu(z):(float)(z/(1+Math.exp(-z)))*v.values[i][t];}return output=mm(down,a);}
    @Override public Matrix backwardGradient(Matrix g){shape(g,output);Matrix da=mm(transpose(down),g);accumulate(gradients.get(1),mm(g,transpose(a)));Matrix du=new Matrix(u.rows(),u.columns()),dv=gate==null?null:new Matrix(u.rows(),u.columns());for(int i=0;i<u.rows();i++)for(int t=0;t<u.columns();t++){float z=u.values[i][t],d=da.values[i][t];if(gate==null)du.values[i][t]=d*geluDerivative(z);else{float s=(float)(1/(1+Math.exp(-z)));du.values[i][t]=d*v.values[i][t]*(s+z*s*(1-s));dv.values[i][t]=d*z*s;}}accumulate(gradients.get(0),mm(du,transpose(input)));Matrix dx=mm(transpose(up),du);if(gate!=null){accumulate(gradients.get(2),mm(dv,transpose(input)));dx=add(dx,mm(transpose(gate),dv));}return dx;}
    @Override public void clearTrainingState(){input=null;u=null;v=null;a=null;output=null;}
}
