# Copyright (c) 2026 Kohei Shintani. Licensed under CC BY-NC 4.0
# (Attribution-NonCommercial): https://creativecommons.org/licenses/by-nc/4.0/
# Commercial use requires prior permission (see LICENSE in the source repository).
# This file has been minified for distribution (comments/docstrings removed).
_D='yeo-johnson'
_C=False
_B=None
_A=1.
import numpy as np
from numpy.linalg import solve,lstsq,cholesky
def r2_score(y_true,y_pred):
	B=y_pred;A=y_true;A=np.asarray(A,float);B=np.asarray(B,float);C=float(np.sum((A-B)**2));E=float(A.mean());D=float(np.sum((A-E)**2))
	if D>0:return _A-C/D
	return _A if C==0 else .0
def mean_squared_error(y_true,y_pred):B=y_pred;A=y_true;A=np.asarray(A,float);B=np.asarray(B,float);return float(np.mean((A-B)**2))
def mean_absolute_error(y_true,y_pred):B=y_pred;A=y_true;A=np.asarray(A,float);B=np.asarray(B,float);return float(np.mean(np.abs(A-B)))
class KFold:
	def __init__(A,n_splits=5,shuffle=_C,random_state=_B):A.n_splits=n_splits;A.shuffle=shuffle;A.random_state=random_state
	def split(A,X,y=_B):
		E=len(X);B=np.arange(E)
		if A.shuffle:np.random.RandomState(A.random_state).shuffle(B)
		C=np.array_split(B,A.n_splits)
		for D in range(A.n_splits):F=C[D];G=np.concatenate([C[A]for A in range(A.n_splits)if A!=D]);yield(G,F)
class StratifiedKFold:
	'各クラス(bin)を n_splits に巡回配分する層化 K-Fold。\n    sklearn とビット同一ではないが「各 fold に各クラスが均等に入る」層化目的は満たす。'
	def __init__(A,n_splits=5,shuffle=_C,random_state=_B):A.n_splits=n_splits;A.shuffle=shuffle;A.random_state=random_state
	def split(A,X,y):
		y=np.asarray(y);E=len(y);F=np.random.RandomState(A.random_state);B=np.empty(E,dtype=int)
		for G in np.unique(y):
			C=np.where(y==G)[0]
			if A.shuffle:F.shuffle(C)
			B[C]=np.arange(len(C))%A.n_splits
		for D in range(A.n_splits):H=np.where(B==D)[0];I=np.where(B!=D)[0];yield(I,H)
class StandardScaler:
	def fit(A,X):X=np.asarray(X,float);A.mean_=X.mean(axis=0);B=X.std(axis=0);A.scale_=np.maximum(B,1e-08);return A
	def transform(A,X):return(np.asarray(X,float)-A.mean_)/A.scale_
	def fit_transform(A,X):return A.fit(X).transform(X)
class RobustScaler:
	'median 中心・IQR スケール（sklearn.RobustScaler 既定と同じ 25-75 パーセンタイル）。'
	def fit(A,X):X=np.asarray(X,float);A.center_=np.median(X,axis=0);B=np.percentile(X,25,axis=0);C=np.percentile(X,75,axis=0);D=C-B;A.scale_=np.maximum(D,1e-08);return A
	def transform(A,X):return(np.asarray(X,float)-A.center_)/A.scale_
	def fit_transform(A,X):return A.fit(X).transform(X)
class PolynomialFeatures:
	'degree=2, include_bias=False。列順を sklearn と一致させる:\n    [x_i (単項, i昇順)] + [x_i*x_j (i<=j, (i,j)昇順)]。'
	def __init__(A,degree=2,include_bias=_C,interaction_only=_C):assert degree==2 and not include_bias;A.interaction_only=interaction_only
	def fit(A,X):A.n_input_=np.asarray(X).shape[1];return A
	def transform(D,X):
		X=np.asarray(X,float);G,A=X.shape;C=[X[:,A]for A in range(A)]
		for B in range(A):
			E=B+1 if D.interaction_only else B
			for F in range(E,A):C.append(X[:,B]*X[:,F])
		return np.column_stack(C)
	def fit_transform(A,X):return A.fit(X).transform(X)
	def get_feature_names_out(F,names):
		A=names;A=list(A);C=len(A);D=list(A)
		for B in range(C):
			G=B+1 if F.interaction_only else B
			for E in range(G,C):D.append(f"{A[B]} {A[E]}"if B!=E else f"{A[B]}^2")
		return np.array(D,dtype=object)
class RidgeCV:
	'中心化 Ridge を αグリッド×K-Fold で選択（切片は平均差で復元）。\n    sklearn.RidgeCV と外部テスト R² が一致することを PoC で確認済み。'
	def __init__(A,alphas,cv=5):A.alphas=list(alphas);A.cv=cv
	def _fit_alpha(G,X,y,a):B=X.mean(0);C=float(y.mean());A=X-B;D=y-C;E=X.shape[1];F=solve(A.T@A+a*np.eye(E),A.T@D);return F,B,C
	def fit(A,X,y):
		X=np.asarray(X,float);y=np.asarray(y,float);H=len(y);F=min(A.cv,H)
		if F<2:C=A.alphas[0];B,D,E=A._fit_alpha(X,y,C);A.coef_=B;A.intercept_=float(E-D@B);A.alpha_=float(C);return A
		O=np.random.RandomState(42).permutation(H);I=np.array_split(O,F);C,J=A.alphas[0],np.inf
		for K in A.alphas:
			G=.0
			for L in range(F):M=I[L];N=np.concatenate([I[A]for A in range(F)if A!=L]);B,D,E=A._fit_alpha(X[N],y[N],K);P=(X[M]-D)@B+E;G+=float(np.sum((y[M]-P)**2))
			if G<J:J,C=G,K
		B,D,E=A._fit_alpha(X,y,C);A.coef_=B;A.intercept_=float(E-D@B);A.alpha_=float(C);return A
	def predict(A,X):return np.asarray(X,float)@A.coef_+A.intercept_
class PowerTransformer:
	def __init__(A,method=_D,standardize=_C):assert method==_D;A.standardize=standardize;A.lambdas_=_B
	@staticmethod
	def _yj(x,lam):
		B=lam;x=np.asarray(x,float);C=np.empty_like(x);A=x>=0
		if abs(B)<1e-06:C[A]=np.log1p(x[A])
		else:C[A]=((x[A]+_A)**B-_A)/B
		if abs(B-2.)<1e-06:C[~A]=-np.log1p(-x[~A])
		else:C[~A]=-((-x[~A]+_A)**(2.-B)-_A)/(2.-B)
		return C
	def _neg_llf(B,lam,x):
		C=len(x);D=B._yj(x,lam);A=float(D.var())
		if A<1e-12:return 1e10
		E=-.5*C*np.log(A)+(lam-_A)*float(np.sum(np.sign(x)*np.log1p(np.abs(x))));return-E
	def _optimize_lambda(E,x):
		A,B=-5.,5.;F=(np.sqrt(5)-1)/2;C=B-F*(B-A);D=A+F*(B-A);G=E._neg_llf(C,x);H=E._neg_llf(D,x)
		for I in range(60):
			if G<H:B,D,H=D,C,G;C=B-F*(B-A);G=E._neg_llf(C,x)
			else:A,C,G=C,D,H;D=A+F*(B-A);H=E._neg_llf(D,x)
			if abs(B-A)<1e-06:break
		return(A+B)/2
	def fit(A,X):X=np.asarray(X,float);A.lambdas_=np.array([A._optimize_lambda(B)for B in X.T]);return A
	def transform(A,X):X=np.asarray(X,float);return np.column_stack([A._yj(X[:,B],A.lambdas_[B])for B in range(X.shape[1])])
	@staticmethod
	def _yj_inv(y,lam):
		B=lam;y=np.asarray(y,float);C=np.empty_like(y);A=y>=0
		if abs(B)<1e-06:C[A]=np.expm1(y[A])
		else:D=np.clip(B*y[A]+_A,1e-12,_B);C[A]=D**(_A/B)-_A
		if abs(B-2.)<1e-06:C[~A]=_A-np.exp(-y[~A])
		else:E=np.clip(-(2.-B)*y[~A]+_A,1e-12,_B);C[~A]=_A-E**(_A/(2.-B))
		return C
	def inverse_transform(A,X):X=np.asarray(X,float);return np.column_stack([A._yj_inv(X[:,B],A.lambdas_[B])for B in range(X.shape[1])])
def permutation_importance(estimator,X,y,n_repeats=5,random_state=42):
	C=n_repeats;B=estimator;X=np.asarray(X,float);y=np.asarray(y,float);I=np.random.RandomState(random_state);J=r2_score(y,B.predict(X));K,D=X.shape;E=np.zeros(D)
	for A in range(D):
		F=np.empty(C)
		for L in range(C):G=X.copy();G[:,A]=X[I.permutation(K),A];F[L]=J-r2_score(y,B.predict(G))
		E[A]=F.mean()
	class M:0
	H=M();H.importances_mean=E;return H
def skew(x):x=np.asarray(x,float);B=x.mean();A=x.std();return .0 if A<1e-12 else float(np.mean(((x-B)/A)**3))
class LightMLP:
	'全結合 NN（relu 隠れ層 + 線形出力）を numpy + Adam で学習。\n    学習安定化のため内部で y を標準化し、最終層の重み・バイアスに畳み込んで\n    predict は変換後 y スケールを直接返す（native C++ predict_mlp と互換）。\n    sklearn.MLPRegressor 互換に coefs_(list of (n_in,n_out)) / intercepts_ / n_iter_ を公開。'
	def __init__(A,hidden_layer_sizes=(64,32),alpha=.0001,max_iter=1500,learning_rate_init=.01,random_state=42):A.hidden=tuple(hidden_layer_sizes);A.alpha=float(alpha);A.max_iter=int(max_iter);A.lr=float(learning_rate_init);A.random_state=random_state
	def fit(B,X,y):
		e=np.random.RandomState(B.random_state);X=np.asarray(X,float);y=np.asarray(y,float);U=float(y.mean());I=float(y.std())
		if I<1e-12:I=_A
		V=(y-U)/I;E=[X.shape[1]]+list(B.hidden)+[1];C=[e.randn(E[A],E[A+1])*np.sqrt(2./E[A])for A in range(len(E)-1)];D=[np.zeros(E[A+1])for A in range(len(E)-1)];M=[np.zeros_like(A)for A in C];N=[np.zeros_like(A)for A in C];O=[np.zeros_like(A)for A in D];P=[np.zeros_like(A)for A in D];F,G,W=.9,.999,1e-08;Y=len(X);f=B.alpha;H=len(C);Z=0;a=np.inf;Q=0;g,h=15,.0001
		for J in range(1,B.max_iter+1):
			Z=J;K=[X];b=[]
			for A in range(H):R=K[-1]@C[A]+D[A];b.append(R);K.append(np.maximum(.0,R)if A<H-1 else R)
			c=K[-1].ravel();d=float(np.mean((c-V)**2))
			if d<a-h:a=d;Q=0
			else:
				Q+=1
				if Q>=g:break
			L=2./Y*(c-V)[:,_B];S=[_B]*H;T=[_B]*H
			for A in reversed(range(H)):
				S[A]=K[A].T@L+f/Y*C[A];T[A]=L.sum(0)
				if A>0:L=L@C[A].T*(b[A-1]>0)
			for A in range(H):M[A]=F*M[A]+(1-F)*S[A];N[A]=G*N[A]+(1-G)*S[A]**2;C[A]-=B.lr*(M[A]/(1-F**J))/(np.sqrt(N[A]/(1-G**J))+W);O[A]=F*O[A]+(1-F)*T[A];P[A]=G*P[A]+(1-G)*T[A]**2;D[A]-=B.lr*(O[A]/(1-F**J))/(np.sqrt(P[A]/(1-G**J))+W)
		C[-1]=C[-1]*I;D[-1]=D[-1]*I+U;B.coefs_=C;B.intercepts_=D;B.n_iter_=Z;return B
	def predict(B,X):
		A=np.asarray(X,float);D=len(B.coefs_)
		for C in range(D):
			A=A@B.coefs_[C]+B.intercepts_[C]
			if C<D-1:A=np.maximum(.0,A)
		return A.ravel()
class LightPipeline:
	"sklearn.Pipeline の最小代替。steps=[(name, step), ...]。最終ステップ以外は\n    fit_transform/transform、最終ステップは fit/predict。pipeline['name'] でアクセス。"
	def __init__(A,steps):B=steps;A.steps=list(B);A._d=dict(B)
	def __getitem__(A,key):return A._d[key]
	def fit(A,X,y=_B):
		B=X
		for(D,C)in A.steps[:-1]:B=C.fit_transform(B)
		A.steps[-1][1].fit(B,y);return A
	def predict(B,X):
		A=X
		for(D,C)in B.steps[:-1]:A=C.transform(A)
		return B.steps[-1][1].predict(A)
def _gp_rbf(Xa,Xb,ls):'ARD-RBF カーネル行列 K[i,j] = exp(-0.5 * Σ_k ((Xa_ik - Xb_jk)/ls_k)^2)。';A=(Xa[:,_B,:]-Xb[_B,:,:])/ls;return np.exp(-.5*(A**2).sum(-1))
def gp_nll_and_grad(params,Xs,yt,d):
	'ARD-RBF + White の負対数周辺尤度と解析勾配。\n    params = [log ls(d), log sigma_var, log noise_var]。Xs:標準化済み, yt:正規化済み y。';C=params;E=len(Xs);L=np.exp(C[:d]).clip(.001,1e2);F=float(np.exp(C[d]).clip(.0001,1e4));G=float(np.exp(C[d+1]).clip(1e-06,1e1));M=Xs[:,_B,:]-Xs[_B,:,:];H=(M/L)**2;I=np.exp(-.5*H.sum(-1));N=F*I+(G+.0001)*np.eye(E)
	try:A=cholesky(N)
	except np.linalg.LinAlgError:return 1e10,np.zeros(d+2)
	D=solve(A.T,solve(A,yt));O=2.*np.log(np.diag(A)).sum();P=float(.5*(yt@D+O));Q=solve(A.T,solve(A,np.eye(E)));J=Q-np.outer(D,D);K=J*(F*I);B=np.empty(d+2);B[:d]=.5*np.einsum('ij,ijk->k',K,H);B[d]=.5*np.sum(K);B[d+1]=.5*G*np.trace(J);return P,B
def minimize_lbfgs(func_grad,x0,max_iter=200,m=10,tol=1e-05):
	'L-BFGS（two-loop recursion + Armijo 線探索）。scipy L-BFGS-B の軽量代替。\n    func_grad(x)->(f,grad)。無制約（境界は func 側 clip で担保）。';R=func_grad;D=np.asarray(x0,float).copy();G,B=R(D);E,C,H=[],[],[]
	for Z in range(max_iter):
		if float(np.linalg.norm(B))<tol:break
		A=B.copy();S=[]
		for(F,L,M)in zip(reversed(E),reversed(C),reversed(H)):I=M*float(F@A);S.append(I);A=A-I*L
		if C:W=float(E[-1]@C[-1])/float(C[-1]@C[-1]);A=A*W
		for(F,L,M,I)in zip(E,C,H,reversed(S)):X=M*float(L@A);A=A+(I-X)*F
		J=-A;N=float(B@J)
		if N>=0:J=-B;N=float(B@J)
		O=_A;Y=.0001;K=D;P=G;Q=B;T=_C
		for a in range(40):
			K=D+O*J;P,Q=R(K)
			if P<=G+Y*O*N:T=True;break
			O*=.5
		if not T:break
		F=K-D;U=Q-B;V=float(F@U)
		if V>1e-10:
			E.append(F);C.append(U);H.append(_A/V)
			if len(E)>m:E.pop(0);C.pop(0);H.pop(0)
		D,G,B=K,P,Q
	return D,G
class LightGP:
	'ARD-RBF ガウス過程回帰（cholesky ソルバ、sklearn 非依存）。\n    ハイパラ (length_scale, sigma_var, noise_var) は外部で最適化して与える。\n    .treg エクスポート用に length_scale/sigma_var/X_train_/alpha_/y_mean_/y_std_ を公開。'
	def __init__(A,length_scale,sigma_var,noise_var):A.length_scale=np.atleast_1d(np.asarray(length_scale,float));A.sigma_var=float(sigma_var);A.noise_var=float(noise_var)
	def fit(A,X,y):X=np.asarray(X,float);y=np.asarray(y,float);A.y_mean_=float(y.mean());B=float(y.std());A.y_std_=B if B>1e-12 else _A;C=(y-A.y_mean_)/A.y_std_;D=len(X);E=A.sigma_var*_gp_rbf(X,X,A.length_scale)+(A.noise_var+.0001)*np.eye(D);A.L_=cholesky(E);A.alpha_=solve(A.L_.T,solve(A.L_,C));A.X_train_=X;return A
	def predict(A,X):
		X=np.asarray(X,float);B=len(X);C=2000
		if B<=C:D=A.sigma_var*_gp_rbf(X,A.X_train_,A.length_scale);return D@A.alpha_*A.y_std_+A.y_mean_
		F=np.empty(B,dtype=float)
		for E in range(0,B,C):G=min(E+C,B);D=A.sigma_var*_gp_rbf(X[E:G],A.X_train_,A.length_scale);F[E:G]=D@A.alpha_*A.y_std_+A.y_mean_
		return F
def nnls(A,b,max_iter=_B):
	'Lawson-Hanson active-set NNLS。戻り値は (x, rnorm) で scipy.optimize.nnls 互換。';F=max_iter;A=np.asarray(A,float);b=np.asarray(b,float);Q,D=A.shape
	if F is _B:F=3*D
	B=np.zeros(D);C=np.zeros(D,bool);G=A.T@(b-A@B);J=0;L=1e-10*max(float(np.abs(A.T@b).max()),1e-300)
	while not C.all()and((~C).any()and G[~C].max()>L):
		J+=1
		if J>F:break
		K=np.where(~C)[0];M=K[np.argmax(G[K])];C[M]=True
		while True:
			N=A[:,C];H=lstsq(N,b,rcond=_B)[0]
			if(H>0).all():B[C]=H;break
			E=np.zeros(D);E[C]=H;I=C&(E<=0);O=(B[I]/(B[I]-E[I])).min();B=B+O*(E-B);C[B<=1e-12]=_C
		G=A.T@(b-A@B)
	P=float(np.linalg.norm(b-A@B));return B,P